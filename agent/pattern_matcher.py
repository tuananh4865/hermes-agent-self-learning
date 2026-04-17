"""
Pattern Matcher — similarity search for proactive failure avoidance.

Given a new task, finds similar past failures from the trajectory index
and generates warnings/hints to help the agent avoid repeating mistakes.

Usage:
    matcher = PatternMatcher()
    result = matcher.match(
        task="refactor the authentication module",
        signals=task_signals
    )
    
    if result.failure_warnings:
        print(result.failure_warnings[0])  # "⚠️ Similar task failed 3x with CONTEXT_OVERFLOW..."
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.failure_classifier import FailureReason, TaskSignals
from agent.trajectory_index import TrajectoryIndex, TrajectoryMatch

logger = logging.getLogger(__name__)


@dataclass
class PatternMatchResult:
    """Result of pattern matching against historical trajectories."""
    similar_failures: List[TrajectoryMatch] = field(default_factory=list)
    similar_successes: List[TrajectoryMatch] = field(default_factory=list)
    failure_warnings: List[str] = field(default_factory=list)
    success_hints: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    recommended_complexity: Optional[str] = None  # upgrade/downgrade based on history


class PatternMatcher:
    """
    Matches new tasks against historical trajectory patterns.
    
    Looks for:
    - Past failures with similar task type/complexity
    - Past successes that might indicate good approaches
    - Generates actionable warnings and hints
    """

    def __init__(self, trajectory_index: Optional[TrajectoryIndex] = None):
        self.index = trajectory_index or TrajectoryIndex()

    def match(
        self,
        task: str,
        signals: TaskSignals,
        min_failures_to_warn: int = 2,
        similarity_threshold: float = 0.3
    ) -> PatternMatchResult:
        """
        Find similar past tasks and generate warnings/hints.
        
        Args:
            task: The user's task description
            signals: Extracted task signals
            min_failures_to_warn: Minimum similar failures before warning
            similarity_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            PatternMatchResult with warnings, hints, recommended actions
        """
        result = PatternMatchResult()
        
        # Find similar past failures
        failures = self.index.find_similar(
            task_message=task,
            task_type=signals.task_type,
            complexity=signals.complexity,
            limit=5,
            include_failed_only=True
        )
        result.similar_failures = [f for f in failures if f.similarity_score >= similarity_threshold]
        
        # Find similar past successes
        successes = self.index.find_similar(
            task_message=task,
            task_type=signals.task_type,
            complexity=signals.complexity,
            limit=3,
            include_failed_only=False
        )
        result.similar_successes = [
            s for s in successes
            if s.completed and s.similarity_score >= similarity_threshold
        ][:3]
        
        # Generate warnings from failures
        if len(result.similar_failures) >= min_failures_to_warn:
            result.failure_warnings = self._generate_failure_warnings(
                result.similar_failures, signals
            )
            result.recommended_actions = self._actions_from_failures(result.similar_failures)
        
        # Generate hints from successes
        if result.similar_successes:
            result.success_hints = self._generate_success_hints(result.similar_successes)
        
        # Adjust complexity based on history
        result.recommended_complexity = self._adjust_complexity(signals, result.similar_failures)
        
        return result

    def _generate_failure_warnings(
        self,
        failures: List[TrajectoryMatch],
        signals: TaskSignals
    ) -> List[str]:
        """Generate human-readable warnings from past failures."""
        warnings = []
        
        # Group failures by reason
        by_reason: Dict[str, List[TrajectoryMatch]] = {}
        for f in failures:
            reason = f.failure_reason or "unknown"
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(f)
        
        for reason, reason_failures in by_reason.items():
            count = len(reason_failures)
            if count < 2:
                continue
            
            try:
                failure_enum = FailureReason(reason)
            except ValueError:
                failure_enum = FailureReason.UNKNOWN
            
            # Format warning based on failure type
            if failure_enum == FailureReason.CONTEXT_OVERFLOW:
                avg_util = sum(f.context_util_final for f in reason_failures) / count
                warnings.append(
                    f"⚠️ Pattern Match: {count} similar tasks failed with CONTEXT_OVERFLOW. "
                    f"Avg context util was {avg_util:.0%}. "
                    f"Recommendation: Compact context early, use subagent research."
                )
            
            elif failure_enum == FailureReason.CONTEXT_DUMMY_ZONE:
                warnings.append(
                    f"⚠️ Pattern Match: {count} similar tasks degraded in quality around "
                    f"40-75% context utilization. Recommendation: Keep context compact, "
                    f"don't let it grow large before checking."
                )
            
            elif failure_enum == FailureReason.TRAJECTORY_CORRECTION_LOOP:
                avg_corrections = sum(f.correction_count for f in reason_failures) / count
                warnings.append(
                    f"⚠️ Pattern Match: {count} similar tasks hit correction loops "
                    f"(avg {avg_corrections:.1f} corrections). "
                    f"Recommendation: Don't iterate on {signals.task_type} logic — "
                    f"spawn a research subagent first to understand the domain."
                )
            
            elif failure_enum == FailureReason.ITERATION_EXHAUSTED:
                avg_tools = sum(f.tool_call_count for f in reason_failures) / count
                warnings.append(
                    f"⚠️ Pattern Match: {count} similar tasks ran out of iterations "
                    f"(avg {avg_tools:.0f} tool calls). "
                    f"Recommendation: Increase iteration budget or decompose the task."
                )
            
            elif failure_enum == FailureReason.TOOL_ERROR_STORM:
                warnings.append(
                    f"⚠️ Pattern Match: {count} similar tasks had high tool error rates. "
                    f"Recommendation: Verify tool availability and permissions first."
                )
            
            elif failure_enum == FailureReason.ARCHITECTURE_MISMATCH:
                warnings.append(
                    f"⚠️ Pattern Match: {count} similar {signals.task_type} tasks "
                    f"failed due to architecture/style mismatches. "
                    f"Recommendation: Study codebase patterns before implementing."
                )
            
            elif failure_enum == FailureReason.MISSING_DEPENDENCY:
                warnings.append(
                    f"⚠️ Pattern Match: {count} similar tasks hit missing dependency issues. "
                    f"Recommendation: Pre-scan imports and dependencies before coding."
                )
            
            elif failure_enum == FailureReason.UNKNOWN:
                warnings.append(
                    f"⚠️ Pattern Match: {count} similar tasks failed for unknown reasons. "
                    f"Approach with caution — consider extra research phase."
                )
        
        return warnings

    def _generate_success_hints(self, successes: List[TrajectoryMatch]) -> List[str]:
        """Generate hints from past successes."""
        hints = []
        
        for s in successes:
            corrections = s.correction_count
            tool_calls = s.tool_call_count
            
            hint_parts = []
            if corrections == 0:
                hint_parts.append("zero corrections")
            elif corrections <= 1:
                hint_parts.append("minimal corrections")
            
            if tool_calls > 0:
                hint_parts.append(f"{tool_calls} tool calls")
            
            hint = f"• Similar task succeeded with {', '.join(hint_parts)}"
            hints.append(hint)
        
        return hints

    def _actions_from_failures(self, failures: List[TrajectoryMatch]) -> List[str]:
        """Infer recommended actions from failure patterns."""
        actions = set()
        
        for f in failures:
            reason = f.failure_reason
            if not reason:
                continue
            
            try:
                failure_enum = FailureReason(reason)
            except ValueError:
                continue
            
            # Map failure types to recommended actions
            action_map = {
                FailureReason.CONTEXT_OVERFLOW: [
                    "compact_at_30pct",
                    "spawn_research_subagent",
                ],
                FailureReason.CONTEXT_DUMMY_ZONE: [
                    "compact_earlier",
                    "reduce_context_growth",
                ],
                FailureReason.TRAJECTORY_CORRECTION_LOOP: [
                    "spawn_research_subagent_first",
                    "require_written_plan",
                ],
                FailureReason.ITERATION_EXHAUSTED: [
                    "increase_iteration_budget",
                    "decompose_task",
                ],
                FailureReason.TOOL_ERROR_STORM: [
                    "check_tool_availability",
                ],
                FailureReason.ARCHITECTURE_MISMATCH: [
                    "include_style_guide",
                    "review_architecture_first",
                ],
                FailureReason.MISSING_DEPENDENCY: [
                    "pre_scan_dependencies",
                ],
            }
            
            if failure_enum in action_map:
                actions.update(action_map[failure_enum])
        
        return list(actions)

    def _adjust_complexity(
        self,
        signals: TaskSignals,
        failures: List[TrajectoryMatch]
    ) -> Optional[str]:
        """Determine if complexity should be adjusted based on failure history."""
        if not failures:
            return None
        
        # Count failures by complexity
        high_failures = sum(1 for f in failures if f.complexity == "high")
        total = len(failures)
        
        # If high complexity tasks consistently fail, suggest downgrade to medium
        if high_failures / total > 0.6 and signals.complexity == "high":
            return "medium"
        
        return None
