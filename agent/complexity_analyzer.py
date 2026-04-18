"""
Complexity Analyzer — determines task complexity to decide if proactive planning is needed.

Uses a two-tier approach:
1. Fast heuristics (no LLM call) — keyword and structural analysis
2. Lightweight LLM classification (optional) — for ambiguous cases

Usage:
    analyzer = ComplexityAnalyzer()
    result = analyzer.analyze(
        task="refactor the entire auth module across 3 services",
        conversation_history=history
    )
    
    if result.complexity in ("medium", "high"):
        planner.spawn_research_subagents(...)
"""

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ComplexityLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ComplexityResult:
    """Result of complexity analysis."""
    complexity: str               # low|medium|high
    confidence: float            # 0.0-1.0
    task_type: str               # refactor|bugfix|migration|etc
    reasoning: str               # Why this complexity was assigned
    signals: Dict[str, Any]      # All extracted signals
    needs_llm_classification: bool = False  # Ambiguous, needs LLM
    needs_proactive_planning: bool = False  # Should spawn subagents


class ComplexityAnalyzer:
    """
    Analyzes task complexity using heuristics + optional LLM + trajectory learning.
    
    Complexity detection triggers:
    - LOW: simple commands, single file, straightforward task
    - MEDIUM: multi-file, requires understanding existing code
    - HIGH: multi-repo, large-scale refactor, architecture work
    
    Learning: Adjusts keyword weights based on past trajectory outcomes.
    """

    # Strong complexity indicators (any of these → high)
    HIGH_INDICATORS = {
        "keywords": [
            "entire", "whole", "full", "complete", "monolith",
            "rearchitect", "redesign", "rewrite from scratch",
            "10+", "dozens", "hundreds of",
            "multiple repos", "across services", "across microservices",
            "brownfield", "legacy codebase", "10 year old",
        ],
        "task_types": ["architecture", "migration", "large_scale_refactor"],
    }

    # Moderate complexity indicators (any → medium)
    MEDIUM_INDICATORS = {
        "keywords": [
            "several", "multiple", "module", "service", "two", "three",
            "across", "integration", "api", "database",
            "3-5", "5-10",
            "feature", "complex", "nontrivial",
        ],
        "task_types": ["refactor", "new_feature"],
    }

    # Simple task indicators (implies low complexity)
    LOW_INDICATORS = {
        "keywords": [
            "simple", "quick", "small", "tiny", "one", "single",
            "just", "easily", "straightforward",
        ],
        "task_types": ["simple", "investigation"],
    }

    # Strong signals for specific complexity
    STRONG_SIGNALS = {
        "file_count": {
            range(10, 1000): "high",
            range(3, 10): "medium",
            range(0, 3): "low",
        },
    }

    def __init__(self, trajectory_index=None, config_weights: Dict[str, List[str]] = None):
        """
        Args:
            trajectory_index: Optional TrajectoryIndex for learning from history.
                              If None, analyzer works in stateless mode.
            config_weights: Optional dict of keyword lists from config to override
                          defaults. Format: {"high": [...], "medium": [...], "low": [...]}.
                          Empty lists or missing keys = use built-in defaults.
        """
        self._trajectory_index = trajectory_index
        # Learned adjustments from past trajectories: {task_type: {"high_kw_boost": 0.1, ...}}
        self._learned_adjustments: Dict[str, Dict[str, float]] = {}

        # Build effective keyword lists: merge config overrides with defaults.
        # Config overrides APPEND to defaults (they don't replace).
        self._keyword_overrides = config_weights or {}
        self._high_keywords = list(self.HIGH_INDICATORS["keywords"])
        self._medium_keywords = list(self.MEDIUM_INDICATORS["keywords"])
        self._low_keywords = list(self.LOW_INDICATORS["keywords"])
        for cat, kw_list in self._keyword_overrides.items():
            if cat == "high" and kw_list:
                self._high_keywords.extend(kw_list)
            elif cat == "medium" and kw_list:
                self._medium_keywords.extend(kw_list)
            elif cat == "low" and kw_list:
                self._low_keywords.extend(kw_list)

    def learn_from_trajectories(self, trajectories: list) -> None:
        """
        Update keyword weights based on past trajectory outcomes.
        
        If a task_type + keyword combo led to failures/high corrections,
        boost the weight of that keyword for future classifications.
        """
        if not trajectories:
            return
        
        # Group by task_type and compute avg correction count
        task_type_stats: Dict[str, Dict[str, float]] = {}
        for t in trajectories:
            tt = getattr(t, "task_type", None) or "unknown"
            if tt not in task_type_stats:
                task_type_stats[tt] = {"corrections": [], "failures": 0, "count": 0}
            task_type_stats[tt]["corrections"].append(getattr(t, "correction_count", 0) or 0)
            if not getattr(t, "completed", True):
                task_type_stats[tt]["failures"] += 1
            task_type_stats[tt]["count"] += 1
        
        # Compute adjustments: if a task_type has high avg corrections or high failure rate,
        # it needs more careful complexity detection (bump up thresholds)
        for tt, stats in task_type_stats.items():
            avg_corrections = sum(stats["corrections"]) / len(stats["corrections"]) if stats["corrections"] else 0
            failure_rate = stats["failures"] / stats["count"] if stats["count"] > 0 else 0
            
            # If avg corrections > 3 or failure rate > 40%, this task_type is harder than expected
            adjustment: Dict[str, float] = {}
            if avg_corrections > 3:
                adjustment["high_kw_boost"] = min(0.3, avg_corrections * 0.05)
            if failure_rate > 0.4:
                adjustment["failure_boost"] = 0.15
            
            if adjustment:
                self._learned_adjustments[tt] = adjustment

    def get_adjustment(self, task_type: str) -> Dict[str, float]:
        """
        Get learned adjustments for a task type.
        
        Checks trajectory_index first (live DB lookup), then falls back to
        the in-memory _learned_adjustments populated by learn_from_trajectories().
        """
        # Try live DB lookup first
        if self._trajectory_index is not None:
            db_adjustment = self._trajectory_index.get_adjustment(task_type)
            if db_adjustment:
                return db_adjustment
        # Fall back to in-memory adjustments
        return self._learned_adjustments.get(task_type, {})

    _LLM_CLASSIFIER_PROMPT = """Given this task, classify its complexity for an AI coding agent.

Task: {task}

Classify as one of:
- LOW: Single file changes, simple fixes, clearly defined scope
- MEDIUM: Multi-file work, needs understanding existing code, moderate scope
- HIGH: Large refactors, multi-repo, architecture changes, unclear scope

Respond with ONLY the complexity level, nothing else. Example response: medium
"""

    def analyze(
        self,
        task: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        force_llm: bool = False
    ) -> ComplexityResult:
        """
        Analyze task complexity.
        
        Args:
            task: The user's task description
            conversation_history: Optional prior conversation for context
            force_llm: Force LLM classification (for testing/debugging)
            
        Returns:
            ComplexityResult with complexity level, confidence, and reasoning
        """
        signals = self._extract_signals(task, conversation_history)
        task_type = signals.get("task_type", "unknown")
        
        # Apply learned adjustments from past trajectories
        adjustment = self.get_adjustment(task_type)
        if adjustment:
            # Boost high keyword count if this task_type historically had failures
            high_kw_boost = adjustment.get("high_kw_boost", 0.0)
            if high_kw_boost > 0 and signals.get("high_keywords", 0) == 0:
                # If there are ANY medium keywords AND this task_type is historically hard,
                # treat it as one high keyword
                if signals.get("medium_keywords", 0) >= 1:
                    signals["_effective_high_kw"] = high_kw_boost
        
        # Try heuristics first
        complexity, confidence, reasoning = self._classify_from_signals(signals)
        
        # If ambiguous (medium confidence), consider LLM classification
        needs_llm = (
            force_llm
            or (confidence < 0.7 and complexity == "medium")
            or self._has_ambiguous_signals(signals)
        )
        
        if needs_llm and complexity == "medium":
            # Could upgrade or stay medium
            llm_result = self._llm_classify(task)
            if llm_result:
                complexity = llm_result
                confidence = 0.8  # LLM is more reliable
                reasoning += f" (refined by LLM to {complexity})"
        
        needs_proactive = complexity in ("medium", "high")
        
        return ComplexityResult(
            complexity=complexity,
            confidence=confidence,
            task_type=signals.get("task_type", "unknown"),
            reasoning=reasoning,
            signals=signals,
            needs_llm_classification=needs_llm,
            needs_proactive_planning=needs_proactive,
        )

    def _extract_signals(
        self,
        task: str,
        history: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Extract all signals from task and conversation history."""
        signals = {}
        task_lower = task.lower()
        
        # Task type
        signals["task_type"] = self._classify_task_type(task_lower)
        
        # Keyword counts
        signals["high_keywords"] = self._count_keywords(task_lower, self._high_keywords)
        signals["medium_keywords"] = self._count_keywords(task_lower, self._medium_keywords)
        signals["low_keywords"] = self._count_keywords(task_lower, self._low_keywords)
        
        # File/directory count hints
        signals["file_hints"] = self._extract_file_hints(task)
        
        # Multi-repo detection
        signals["multi_repo"] = self._detect_multi_repo(task_lower)
        
        # Stack trace / error dump (implies bugfix/investigation)
        signals["has_stack_trace"] = self._has_stack_trace(task)
        
        # Test mention
        signals["mentions_test"] = "test" in task_lower
        
        # Duration hints
        signals["duration_hints"] = self._extract_duration_hints(task)
        
        # Prior context from history
        if history:
            signals["has_extensive_history"] = len(history) > 20
            signals["prior_complex_work"] = self._had_complex_work(history)
        
        return signals

    def _classify_task_type(self, text: str) -> str:
        """Classify the type of task."""
        type_keywords = {
            "refactor": ["refactor", "restructure", "reorganize", "clean up", "improve code"],
            "bugfix": ["bug", "fix", "crash", "error", "broken", "issue", "patch"],
            "migration": ["migrate", "upgrade", "port", "convert to", "move to"],
            "new_feature": ["add", "implement", "new", "create", "build", "feature"],
            "investigation": ["find", "search", "investigate", "debug", "trace", "understand"],
            "architecture": ["architect", "design pattern", "system design", "structure"],
            "simple": ["change", "update", "rename", "move", "delete", "remove"],
        }
        
        for task_type, keywords in type_keywords.items():
            if any(kw in text for kw in keywords):
                return task_type
        
        return "unknown"

    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        """Count how many keywords appear in text."""
        return sum(1 for kw in keywords if kw in text)

    def _extract_file_hints(self, task: str) -> int:
        """Extract file count hints from task description."""
        # Look for patterns like "5 files", "3 modules", "10+ files"
        patterns = [
            r'(\d+)\+?\s*(?:files?|modules?|services?|components?)',
            r'(?:entire|whole|all)\s+(?:the\s+)?([\w/]+)',
            r'modules?/([\w/]+)',
        ]
        
        total = 0
        for pattern in patterns:
            matches = re.findall(pattern, task.lower())
            for m in matches:
                try:
                    val = int(m)
                    if 1 <= val <= 100:
                        total += val
                except (ValueError, TypeError):
                    pass
        
        return total

    def _detect_multi_repo(self, text: str) -> bool:
        """Detect if task spans multiple repositories."""
        indicators = [
            "frontend/", "backend/", "frontend", "backend",
            "shared/", "common/", "libs/", "packages/",
            "micro-frontend", "microservices",
            "repo", "repository",
        ]
        
        matches = [ind for ind in indicators if ind in text]
        # Multiple indicators or explicit multi-repo mention
        return len(matches) >= 2 or "multiple repos" in text

    def _has_stack_trace(self, task: str) -> bool:
        """Detect if task contains a stack trace (suggests bugfix)."""
        stack_indicators = ["traceback", "exception", "error:", "at ", "line ", "in "]
        lines = task.split('\n')
        if len(lines) > 5:
            # Looks like a stack trace
            if sum(1 for line in lines[:10] if any(ind in line for ind in stack_indicators)) >= 3:
                return True
        return False

    def _extract_duration_hints(self, task: str) -> Optional[str]:
        """Extract duration/effort hints from task."""
        duration_patterns = [
            (r'(\d+)\s*hours?', 'hours'),
            (r'(\d+)\s*days?', 'days'),
            (r'7\s*hours', '7h'),
            (r'35,?000\s*lines', '35k_lines'),
        ]
        
        for pattern, label in duration_patterns:
            if re.search(pattern, task.lower()):
                return label
        
        return None

    def _had_complex_work(self, history: List[Dict[str, Any]]) -> bool:
        """Check if conversation history shows prior complex work."""
        if len(history) > 30:
            return True
        
        # Check for many tool calls
        tool_calls = sum(
            1 for m in history
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        
        return tool_calls > 20

    def _has_ambiguous_signals(self, signals: Dict[str, Any]) -> bool:
        """Check if signals are mixed enough to warrant LLM classification."""
        high = signals.get("high_keywords", 0)
        medium = signals.get("medium_keywords", 0)
        low = signals.get("low_keywords", 0)
        
        # Mixed signals: all categories have similar counts
        total = high + medium + low
        if total == 0:
            return True
        
        # If the dominant category isn't clear, ambiguous
        max_count = max(high, medium, low)
        return max_count / total < 0.6

    def _classify_from_signals(
        self, signals: Dict[str, Any]
    ) -> tuple[str, float, str]:
        """Classify complexity from extracted signals."""
        high_kw = signals.get("high_keywords", 0)
        # Include learned boost from trajectory history
        effective_high_kw = high_kw + int(signals.get("_effective_high_kw", 0))
        med_kw = signals.get("medium_keywords", 0)
        low_kw = signals.get("low_keywords", 0)
        file_hints = signals.get("file_hints", 0)
        multi_repo = signals.get("multi_repo", False)
        has_stack = signals.get("has_stack_trace", False)
        task_type = signals.get("task_type", "unknown")
        
        # Strong signals for HIGH
        if multi_repo:
            return "high", 0.95, "Multi-repo task detected"
        
        if file_hints >= 10:
            return "high", 0.9, f"File hint suggests {file_hints} files"
        
        if effective_high_kw >= 2:
            learned = effective_high_kw - high_kw
            if learned > 0:
                reason = f"Multiple high-complexity keywords ({effective_high_kw}, including {learned:.1f} from learned history)"
            else:
                reason = f"Multiple high-complexity keywords ({effective_high_kw})"
            return "high", 0.85, reason
        
        if task_type == "architecture":
            return "high", 0.9, "Architecture task"
        
        # Medium signals
        if med_kw >= 2 and high_kw == 0:
            return "medium", 0.7, f"Multiple medium-complexity keywords ({med_kw})"
        
        if file_hints >= 3:
            return "medium", 0.75, f"File hint suggests {file_hints} files"
        
        if task_type in ("refactor", "new_feature"):
            if med_kw >= 1 or file_hints >= 2:
                return "medium", 0.7, f"{task_type} task with moderate scope"
        
        # Learned boost: if task_type historically had failures, upgrade medium→high
        if effective_high_kw > high_kw and med_kw >= 1:
            return "medium", 0.65, f"Upgraded from medium due to learned history for {task_type}"
        
        # Strong signals for LOW
        if low_kw >= 2 and med_kw == 0 and high_kw == 0:
            return "low", 0.9, "Simple task indicators"
        
        if task_type == "simple":
            return "low", 0.8, "Simple task type"
        
        if file_hints <= 1 and med_kw <= 1 and high_kw == 0:
            return "low", 0.65, "Minimal complexity indicators"
        
        # Stack trace usually means bugfix which is often medium
        if has_stack and task_type == "bugfix":
            return "medium", 0.6, "Bugfix with stack trace"
        
        # Default to medium-low if unclear
        return "medium", 0.5, "Insufficient signals for clear classification"

    def _llm_classify(self, task: str) -> Optional[str]:
        """
        Use LLM for classification (optional, for ambiguous cases).
        
        Returns None if LLM classification fails or isn't configured.
        """
        # This would make an LLM call. For now, return None
        # as LLM classification requires provider access.
        # The caller should implement actual LLM call if needed.
        return None
