"""
Proactive Planner — orchestrates subagent spawning and context injection for complex tasks.

Combines complexity analysis + pattern matching to proactively spawn research
subagents BEFORE the main agent starts working, injecting findings as context.

Usage:
    planner = ProactivePlanner()
    
    result = planner.plan_ahead(
        task="refactor the authentication system across 3 services",
        conversation_history=[],
        parent_agent=agent
    )
    
    if result.should_proceed:
        # result.injected_context contains research findings
        # Agent can now proceed with enriched context
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent.complexity_analyzer import ComplexityAnalyzer, ComplexityResult
from agent.failure_classifier import TaskSignals
from agent.pattern_matcher import PatternMatcher, PatternMatchResult

logger = logging.getLogger(__name__)


@dataclass
class SubagentResult:
    """Result from a spawned research subagent."""
    subagent_id: str
    goal: str
    success: bool
    findings: str  # Compressed findings to inject
    files_found: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ProactivePlanResult:
    """Result of proactive planning phase."""
    should_proceed: bool            # True if agent should continue, False to abort
    complexity: str                 # detected complexity level
    injected_context: str           # Research context to inject
    pattern_warnings: List[str]     # Failure pattern warnings
    recommended_actions: List[str]   # From pattern matcher
    spawned_subagents: List[str]     # IDs of spawned subagents
    adjusted_iteration_budget: Optional[int]  # Adjusted if needed
    abort_reason: Optional[str]      # Why planning was aborted
    reasoning: str                  # Human-readable reasoning


class ProactivePlanner:
    """
    Orchestrates proactive planning for complex tasks.
    
    Combines:
    - Complexity analysis (should we plan ahead?)
    - Pattern matching (what failed before?)
    - Subagent spawning (research before coding)
    - Context injection (pass findings to main agent)
    """

    def __init__(
        self,
        complexity_analyzer: Optional[ComplexityAnalyzer] = None,
        pattern_matcher: Optional[PatternMatcher] = None,
        enabled: bool = True,
        complexity_threshold: str = "medium",
        max_subagents: int = 3,
        subagent_timeout: int = 120,
        min_failures_to_warn: int = 2,
    ):
        """
        Initialize the proactive planner.
        
        Args:
            enabled: Whether proactive planning is enabled
            complexity_threshold: Minimum complexity to trigger planning
            max_subagents: Maximum subagents to spawn per task
            subagent_timeout: Timeout per subagent in seconds
            min_failures_to_warn: Min similar failures before warning
        """
        self.enabled = enabled
        self.complexity_threshold = complexity_threshold
        self.max_subagents = max_subagents
        self.subagent_timeout = subagent_timeout
        self.min_failures_to_warn = min_failures_to_warn
        
        self.complexity_analyzer = complexity_analyzer or ComplexityAnalyzer()
        self.pattern_matcher = pattern_matcher or PatternMatcher()

    def plan_ahead(
        self,
        task: str,
        conversation_history: List[Dict[str, Any]],
        parent_agent,
        signals: Optional[TaskSignals] = None,
        complexity_override: Optional[str] = None,
    ) -> ProactivePlanResult:
        """
        Analyze task and optionally spawn research subagents.
        
        Args:
            task: The user's task description
            conversation_history: Prior conversation messages
            parent_agent: The parent AIAgent instance
            signals: Pre-extracted task signals (optional)
            complexity_override: Force a complexity level (optional)
            
        Returns:
            ProactivePlanResult with decision and injected context
        """
        if not self.enabled:
            return ProactivePlanResult(
                should_proceed=True,
                complexity="low",
                injected_context="",
                pattern_warnings=[],
                recommended_actions=[],
                spawned_subagents=[],
                reasoning="Proactive planning disabled"
            )

        # Step 1: Complexity analysis
        complexity_result = self.complexity_analyzer.analyze(
            task=task,
            conversation_history=conversation_history
        )
        
        complexity = complexity_override or complexity_result.complexity
        
        # Check if complexity meets threshold
        if not self._meets_threshold(complexity):
            return ProactivePlanResult(
                should_proceed=True,
                complexity=complexity,
                injected_context="",
                pattern_warnings=[],
                recommended_actions=[],
                spawned_subagents=[],
                reasoning=f"Complexity '{complexity}' below threshold '{self.complexity_threshold}'"
            )

        # Step 2: Pattern matching against failures
        task_signals = signals or self._signals_from_complexity(complexity_result)
        patterns = self.pattern_matcher.match(
            task=task,
            signals=task_signals,
            min_failures_to_warn=self.min_failures_to_warn,
        )

        # Step 3: Build initial context from patterns (before spawning)
        initial_context = self._format_pattern_context(patterns)
        
        if patterns.failure_warnings and len(patterns.similar_failures) >= self.min_failures_to_warn:
            # Known failure pattern — be extra cautious
            logger.info(
                "ProactivePlanner: %d similar failures detected for task type '%s'",
                len(patterns.similar_failures), complexity_result.task_type
            )

        # Step 4: Spawn research subagents if needed
        spawned = []
        injected_context = initial_context
        
        if complexity in ("medium", "high") or patterns.recommended_actions:
            spawned = self._spawn_research_subagents(
                task=task,
                complexity=complexity,
                patterns=patterns,
                parent_agent=parent_agent,
            )
            
            if spawned:
                # Collect findings from all subagents
                findings_list = [s.findings for s in spawned if s.success and s.findings]
                if findings_list:
                    injected_context = self._format_research_context(
                        findings_list, patterns
                    )

        # Step 5: Determine if we should proceed
        should_proceed = True
        abort_reason = None
        
        # Check if task was aborted before due to similar failures
        # (only abort if pattern matcher found very similar failures)
        if len(patterns.similar_failures) >= 3:
            failure_reasons = set(f.failure_reason for f in patterns.similar_failures)
            if FailureReason.CONTEXT_OVERFLOW in failure_reasons:
                should_proceed = True  # We'll just compact earlier
            # Could add more abort conditions here

        # Step 6: Adjust iteration budget based on history
        adjusted_budget = self._adjust_budget(
            complexity, patterns, parent_agent.max_iterations
        )

        return ProactivePlanResult(
            should_proceed=should_proceed,
            complexity=complexity,
            injected_context=injected_context,
            pattern_warnings=patterns.failure_warnings,
            recommended_actions=patterns.recommended_actions,
            spawned_subagents=[s.subagent_id for s in spawned],
            adjusted_iteration_budget=adjusted_budget,
            abort_reason=abort_reason,
            reasoning=(
                f"Complexity '{complexity}' ({complexity_result.reasoning}). "
                f"Matched {len(patterns.similar_failures)} similar failures, "
                f"{len(patterns.similar_successes)} similar successes. "
                f"Spawned {len(spawned)} research subagents."
            ),
        )

    def _meets_threshold(self, complexity: str) -> bool:
        """Check if complexity meets the configured threshold."""
        thresholds = {"low": 0, "medium": 1, "high": 2}
        task_level = thresholds.get(complexity, 0)
        min_level = thresholds.get(self.complexity_threshold, 1)
        return task_level >= min_level

    def _signals_from_complexity(self, result: ComplexityResult) -> TaskSignals:
        """Convert ComplexityResult to TaskSignals for pattern matcher."""
        return TaskSignals(
            task_type=result.task_type,
            complexity=result.complexity,
            codebase_size="medium" if result.complexity == "high" else "small",
        )

    def _spawn_research_subagents(
        self,
        task: str,
        complexity: str,
        patterns: PatternMatchResult,
        parent_agent,
    ) -> List[SubagentResult]:
        """
        Spawn parallel research subagents targeting failure patterns.
        
        Subagent types:
        - File finder: Find relevant files for the task
        - Dependency scanner: Find imports and dependencies
        - Test finder: Find related tests
        """
        from agent.failure_classifier import FailureReason
        
        results = []
        
        # Determine which research tasks to spawn based on patterns
        research_tasks = []
        
        # Always spawn file finder for medium+ tasks
        if complexity == "high":
            research_tasks.append({
                "id": "file_finder",
                "goal": f"Find all files relevant to this task: '{task}'. Search for relevant source files, configuration, and related modules. Return a list of file paths with brief descriptions of what each contains.",
                "max_iterations": 15,
            })
            research_tasks.append({
                "id": "test_finder", 
                "goal": f"Find all test files related to: '{task}'. Look for test directories, test files, and mock/stub files. Return file paths with what they test.",
                "max_iterations": 10,
            })
        
        # Add dependency research if MISSING_DEPENDENCY was a pattern
        has_dep_pattern = any(
            f.failure_reason == FailureReason.MISSING_DEPENDENCY.value
            for f in patterns.similar_failures
        )
        if has_dep_pattern or complexity == "high":
            research_tasks.append({
                "id": "dep_scanner",
                "goal": f"For task '{task}', scan the codebase to find all imports, dependencies, and external packages used. Return a summary of the dependency graph.",
                "max_iterations": 8,
            })
        
        # Limit subagents
        research_tasks = research_tasks[:self.max_subagents]
        
        if not research_tasks:
            return results
        
        # Spawn subagents (in parallel via delegate_task)
        # Note: parent_agent must have delegate_task available
        if not hasattr(parent_agent, 'delegate_task'):
            logger.warning("ProactivePlanner: parent_agent has no delegate_task method")
            return results
        
        # For now, spawn sequentially (batch would require concurrent execution)
        for task_def in research_tasks:
            result = self._run_research_subagent(
                task_id=task_def["id"],
                goal=task_def["goal"],
                max_iterations=task_def["max_iterations"],
                parent_agent=parent_agent,
            )
            results.append(result)
        
        return results

    def _run_research_subagent(
        self,
        task_id: str,
        goal: str,
        max_iterations: int,
        parent_agent,
    ) -> SubagentResult:
        """
        Run a single research subagent.
        
        Returns SubagentResult with findings.
        """
        try:
            # Call delegate_task on parent agent
            result = parent_agent.delegate_task(
                goal=goal,
                context=f"You are a research subagent. Be thorough but concise. Return your findings as a structured summary.",
                toolsets=["terminal", "file"],
                max_iterations=max_iterations,
            )
            
            return SubagentResult(
                subagent_id=task_id,
                goal=goal,
                success=True,
                findings=str(result)[:2000] if result else "",  # Limit size
            )
            
        except Exception as e:
            logger.warning("ProactivePlanner: subagent %s failed: %s", task_id, e)
            return SubagentResult(
                subagent_id=task_id,
                goal=goal,
                success=False,
                findings="",
                error=str(e)[:200],
            )

    def _format_pattern_context(self, patterns: PatternMatchResult) -> str:
        """Format failure pattern warnings into a context block."""
        if not patterns.failure_warnings:
            return ""
        
        lines = [
            "\n\n╔══════════════════════════════════════════════════════════╗",
            "║  PATTERN MATCH WARNING (from past failures)              ║",
            "╠══════════════════════════════════════════════════════════╣",
        ]
        
        for warning in patterns.failure_warnings:
            lines.append(f"║ {warning[:55]}")
        
        lines.extend([
            "║                                                          ║",
            "║  Recommended actions:",
        ])
        
        for action in patterns.recommended_actions[:5]:
            lines.append(f"║    • {action}")
        
        lines.extend([
            "╚══════════════════════════════════════════════════════════╝\n",
        ])
        
        return "\n".join(lines)

    def _format_research_context(
        self,
        findings_list: List[str],
        patterns: PatternMatchResult
    ) -> str:
        """Format research subagent findings into an injected context block."""
        lines = [
            "\n\n╔══════════════════════════════════════════════════════════╗",
            "║  RESEARCH CONTEXT (auto-generated by proactive planner) ║",
            "╠══════════════════════════════════════════════════════════╣",
        ]
        
        for findings in findings_list:
            if findings:
                # Indent and wrap
                wrapped = self._wrap_text(findings[:1500], width=56)
                for line in wrapped:
                    lines.append(f"║  {line}")
        
        if patterns.failure_warnings:
            lines.append("║                                                          ║")
            lines.append("║  ⚠️  Past failure warnings:")
            for warning in patterns.failure_warnings[:2]:
                wrapped = self._wrap_text(warning, width=56)
                for line in wrapped[:2]:
                    lines.append(f"║     {line}")
        
        lines.extend([
            "╚══════════════════════════════════════════════════════════╝\n",
        ])
        
        return "\n".join(lines)

    def _adjust_budget(
        self,
        complexity: str,
        patterns: PatternMatchResult,
        current_budget: int
    ) -> Optional[int]:
        """Adjust iteration budget based on complexity and history."""
        # High complexity tasks might need more iterations
        if complexity == "high":
            return min(current_budget + 30, 150)
        
        # If past similar tasks exhausted budget, give more
        if patterns.similar_failures:
            avg_iterations = sum(
                getattr(f, 'tool_call_count', 0) for f in patterns.similar_failures
            ) / len(patterns.similar_failures)
            
            if avg_iterations > current_budget * 0.8:
                return min(current_budget + 20, 120)
        
        return None

    @staticmethod
    def _wrap_text(text: str, width: int = 56) -> List[str]:
        """Simple word wrap for context formatting."""
        words = text.split()
        lines = []
        current = ""
        
        for word in words:
            if len(current) + len(word) + 1 <= width:
                current += (" " if current else "") + word
            else:
                if current:
                    lines.append(current)
                current = word
        
        if current:
            lines.append(current)
        
        return lines


# Import FailureReason at module level for type checking
from agent.failure_classifier import FailureReason
