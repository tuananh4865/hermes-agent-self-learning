"""
Outcome Evaluator — post-task evaluation to close the self-learning loop.

Evaluates task outcomes at the end of each conversation turn, determines
SUCCESS/FAILURE/PARTIAL_SUCCESS, and feeds results to the trajectory index.

Usage:
    evaluator = OutcomeEvaluator()
    report = evaluator.evaluate(
        messages=conversation_messages,
        exit_reason="iteration_exhausted",
        signals=task_signals,
        planned_actions=["spawn_research", "compact_early"]
    )
    
    if report.outcome == Outcome.PARTIAL_SUCCESS:
        # index partial success too
"""

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.failure_classifier import FailureClassifier, FailureReason, TaskSignals


class Outcome(enum.Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


@dataclass
class OutcomeReport:
    """Result of post-task outcome evaluation."""
    outcome: Outcome
    failure_reason: Optional[FailureReason]
    correction_count: int
    was_suspicious: bool         # model claimed done but might have missed
    lessons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    signals: Optional[TaskSignals] = None


class OutcomeEvaluator:
    """
    Evaluates task outcomes and determines lessons learned.
    
    This closes the self-learning loop:
        Task completes → evaluate → index → future tasks benefit
    """

    def __init__(self):
        self.classifier = FailureClassifier()

    def evaluate(
        self,
        messages: List[Dict[str, Any]],
        exit_reason: str,
        signals: Optional[TaskSignals] = None,
        planned_actions: Optional[List[str]] = None,
        usage_stats: Optional[Dict[str, Any]] = None
    ) -> OutcomeReport:
        """
        Evaluate the outcome of a completed task.
        
        Args:
            messages: Full conversation message history
            exit_reason: Why the conversation ended
            signals: Pre-extracted task signals (optional, will extract if not provided)
            planned_actions: Actions the proactive planner took (for learning)
            usage_stats: API usage stats (token counts, etc.)
            
        Returns:
            OutcomeReport with outcome, failure reason, lessons
        """
        # Extract signals if not provided
        if signals is None:
            signals = self.classifier.extract_signals(messages, exit_reason, usage_stats)
        
        # Determine outcome
        outcome, failure_reason = self._determine_outcome(
            messages, exit_reason, signals
        )
        
        # Detect suspicious completions
        suspicious = self._detect_suspicious_completion(messages)
        
        # Extract lessons
        lessons = self._extract_lessons(outcome, failure_reason, signals, planned_actions)
        
        # Generate recommendations for future
        recommendations = self._generate_recommendations(
            outcome, failure_reason, signals
        )
        
        return OutcomeReport(
            outcome=outcome,
            failure_reason=failure_reason,
            correction_count=signals.correction_count,
            was_suspicious=suspicious,
            lessons=lessons,
            recommendations=recommendations,
            signals=signals,
        )

    def _determine_outcome(
        self,
        messages: List[Dict[str, Any]],
        exit_reason: str,
        signals: TaskSignals
    ) -> tuple[Outcome, Optional[FailureReason]]:
        """Determine if task was successful, partial, or failed."""
        
        # Explicit failure reasons
        if exit_reason in ("iteration_exhausted", "budget_exhausted", "error"):
            reason = self.classifier.classify(messages, signals, exit_reason)
            return Outcome.FAILURE, reason
        
        # Check for suspicious completion
        if self._detect_suspicious_completion(messages):
            return Outcome.PARTIAL_SUCCESS, FailureReason.SLOP_GENERATED
        
        # High correction count = partial success (model kept needing fixes)
        if signals.correction_count >= 3:
            reason = self.classifier.classify(messages, signals, exit_reason)
            return Outcome.PARTIAL_SUCCESS, reason
        
        # High error density
        if signals.error_density > 0.2:
            return Outcome.PARTIAL_SUCCESS, FailureReason.TOOL_ERROR_STORM
        
        # Context in dumb zone at end
        if signals.context_util_final > 0.80:
            reason = self.classifier.classify(messages, signals, exit_reason)
            if reason != FailureReason.SUCCESS:
                return Outcome.PARTIAL_SUCCESS, reason
        
        # Success: clean exit, low corrections, reasonable tool usage
        if exit_reason == "completed" and signals.correction_count <= 1:
            return Outcome.SUCCESS, None
        
        # Default: partial success
        reason = self.classifier.classify(messages, signals, exit_reason)
        return Outcome.PARTIAL_SUCCESS, reason

    def _detect_suspicious_completion(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Detect if model claimed to complete but likely missed requirements.
        
        Signs:
        - Model says "done" but has tool_calls that weren't executed
        - Model summarizes work but doesn't match actual changes
        - Very few tool calls for a complex task
        """
        if not messages:
            return False
        
        last_msg = messages[-1]
        if last_msg.get("role") != "assistant":
            return False
        
        content = str(last_msg.get("content", "")).lower()
        
        # Model said it's done
        done_phrases = [
            "i've completed", "i've finished", "i have completed",
            "i have finished", "all done", "that's all",
            "here's the", "the implementation is complete",
        ]
        said_done = any(phrase in content for phrase in done_phrases)
        
        if not said_done:
            return False
        
        # Count how many actual file modifications happened
        file_mods = 0
        for msg in messages:
            if msg.get("role") == "tool":
                content = str(msg.get("content", "")).lower()
                if any(kw in content for kw in ["created", "modified", "updated", "written"]):
                    file_mods += 1
        
        # If model said done but made almost no changes, suspicious
        if file_mods < 2:
            # Check if this was supposed to be a complex task
            for msg in messages:
                if msg.get("role") == "user":
                    user_text = str(msg.get("content", "")).lower()
                    if any(kw in user_text for kw in [
                        "implement", "create", "add feature", "build",
                        "multiple", "several files"
                    ]):
                        return True
        
        return False

    def _extract_lessons(
        self,
        outcome: Outcome,
        failure_reason: Optional[FailureReason],
        signals: TaskSignals,
        planned_actions: Optional[List[str]] = None
    ) -> List[str]:
        """Extract actionable lessons from this task run."""
        lessons = []
        
        if outcome == Outcome.SUCCESS:
            lessons.append(f"Task type '{signals.task_type}' succeeded with complexity '{signals.complexity}'")
            if signals.correction_count == 0:
                lessons.append("Zero-correction completion — model understood requirements correctly")
            return lessons
        
        if outcome == Outcome.PARTIAL_SUCCESS:
            if signals.correction_count >= 3:
                lessons.append(f"High correction count ({signals.correction_count}) suggests unclear requirements")
            
            if failure_reason == FailureReason.CONTEXT_DUMMY_ZONE:
                lessons.append(f"Context at {signals.context_util_final:.0%} — should compact earlier")
            
            if failure_reason == FailureReason.TOOL_ERROR_STORM:
                lessons.append(f"High error density ({signals.error_density:.0%}) — tool usage strategy needs review")
        
        if outcome == Outcome.FAILURE:
            if failure_reason == FailureReason.CONTEXT_OVERFLOW:
                lessons.append("Context overflow — needs subagent strategy or earlier compaction")
            
            elif failure_reason == FailureReason.ITERATION_EXHAUSTED:
                lessons.append(f"Iteration exhaustion with {signals.iteration_count} turns — increase budget or decompose task")
            
            elif failure_reason == FailureReason.TRAJECTORY_CORRECTION_LOOP:
                lessons.append("Correction loop detected — task needs better research phase")
            
            elif failure_reason == FailureReason.MISSING_DEPENDENCY:
                lessons.append("Missing dependency — should pre-scan imports before coding")
        
        # Check if proactive planning helped
        if planned_actions:
            if "spawn_research" in planned_actions and outcome == Outcome.SUCCESS:
                lessons.append("Proactive research subagents helped")
        
        return lessons

    def _generate_recommendations(
        self,
        outcome: Outcome,
        failure_reason: Optional[FailureReason],
        signals: TaskSignals
    ) -> List[str]:
        """Generate recommendations for future similar tasks."""
        recommendations = []
        
        # Failure-type specific recommendations
        if failure_reason == FailureReason.CONTEXT_OVERFLOW:
            recommendations.append("compact_at_30pct")
            recommendations.append("use_subagent_research")
        
        elif failure_reason == FailureReason.CONTEXT_DUMMY_ZONE:
            recommendations.append("compact_earlier")
        
        elif failure_reason == FailureReason.TRAJECTORY_CORRECTION_LOOP:
            recommendations.append("spawn_research_subagent_first")
            recommendations.append("require_written_plan")
        
        elif failure_reason == FailureReason.ITERATION_EXHAUSTED:
            if signals.complexity == "high":
                recommendations.append("increase_iteration_budget")
            else:
                recommendations.append("decompose_task")
        
        elif failure_reason == FailureReason.MISSING_DEPENDENCY:
            recommendations.append("pre_scan_dependencies")
        
        elif failure_reason == FailureReason.ARCHITECTURE_MISMATCH:
            recommendations.append("include_style_guide")
            recommendations.append("review_architecture_first")
        
        # Complexity-based recommendations
        if signals.complexity == "high":
            if "spawn_research" not in recommendations:
                recommendations.append("spawn_research_subagent_first")
        
        if signals.multi_repo:
            if "decompose_task" not in recommendations:
                recommendations.append("per_repo_subagents")
        
        return list(set(recommendations))  # Dedupe
