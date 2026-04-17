"""
Failure Classifier — analyzes conversation trajectories to determine WHY a task failed.

Extracts meaningful signals from the message history and classifies the root cause
into one of the FailureReason enums. This feeds the trajectory index for
proactive self-healing: similar tasks in the future can be warned or handled
with adjusted strategy.
"""

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class FailureReason(enum.Enum):
    # Context failures
    CONTEXT_OVERFLOW = "context_overflow"           # context window exhausted
    CONTEXT_DUMMY_ZONE = "context_dummy_zone"       # >40% util, quality degraded
    CONTEXT_STALE = "context_stale"                 # outdated info in context

    # Trajectory failures
    TRAJECTORY_CORRECTION_LOOP = "trajectory_correction_loop"  # model keeps being corrected
    TRAJECTORY_WRONG_DIRECTION = "trajectory_wrong_direction"  # understood problem wrong
    TRAJECTORY_SELF_INTERFERENCE = "trajectory_self_interference"  # model contradicts itself

    # Execution failures
    ITERATION_EXHAUSTED = "iteration_exhausted"     # ran out of tool-call iterations
    TOOL_ERROR_STORM = "tool_error_storm"           # repeated tool errors
    FILE_ACCESS_CONFLICT = "file_access_conflict"   # concurrent file modification

    # Quality failures
    SLOP_GENERATED = "slop_generated"               # output passes but fails review
    ARCHITECTURE_MISMATCH = "architecture_mismatch"  # solution doesn't fit codebase style
    MISSING_DEPENDENCY = "missing_dependency"       # assumed wrong dependencies

    # Unknown / no failure
    UNKNOWN = "unknown"
    SUCCESS = "success"


@dataclass
class TaskSignals:
    """High-level signals extracted from a conversation."""
    task_type: str = "unknown"          # refactor|bugfix|migration|new_feature|investigation|simple
    complexity: str = "low"             # low|medium|high
    codebase_size: str = "small"        # small|medium|large
    files_affected: int = 0
    multi_repo: bool = False
    correction_count: int = 0           # how many times user corrected the model
    tool_call_count: int = 0
    context_util_final: float = 0.0     # final context utilization (0.0-1.0)
    has_tests: bool = False
    error_density: float = 0.0         # tool errors per tool call
    iteration_count: int = 0
    duration_seconds: int = 0
    exit_reason: str = ""
    error_messages: List[str] = field(default_factory=list)


class FailureClassifier:
    """
    Analyzes a conversation to determine the root cause of failure.
    
    Usage:
        classifier = FailureClassifier()
        signals = classifier.extract_signals(messages, exit_reason, usage_stats)
        reason = classifier.classify(messages, signals, exit_reason)
    """

    # Keywords for task type classification
    TASK_TYPE_KEYWORDS = {
        "refactor": ["refactor", "restructure", "reorganize", "rewrite", "clean up", "improve"],
        "bugfix": ["bug", "fix", "crash", "error", "broken", "issue", "patch"],
        "migration": ["migrate", "migration", "upgrade", "port", "convert to", "move to"],
        "new_feature": ["add", "implement", "new", "create", "build", "feature"],
        "investigation": ["find", "search", "investigate", "debug", "trace", "understand", "how does"],
        "architecture": ["architect", "design", "system", "pattern", "structure"],
    }

    # Complexity heuristics (fast, no LLM needed)
    COMPLEXITY_KEYWORDS = {
        "high": ["entire", "whole", "monolith", "10+", "multiple repos", "across"],
        "medium": ["several", "module", "service", "3-5", "two"],
    }

    def extract_signals(
        self,
        messages: List[Dict[str, Any]],
        exit_reason: str,
        usage_stats: Optional[Dict[str, Any]] = None
    ) -> TaskSignals:
        """
        Extract high-level signals from conversation messages.
        
        Args:
            messages: Full message history (user, assistant, tool messages)
            exit_reason: Why the conversation ended (e.g. "iteration_exhausted", "completed")
            usage_stats: Optional API usage stats (token counts, etc.)
        """
        signals = TaskSignals()
        
        # Flatten all text for keyword analysis
        all_text = self._flatten_messages(messages)
        
        # Task type
        signals.task_type = self._classify_task_type(all_text)
        
        # Complexity
        signals.complexity = self._classify_complexity(all_text)
        
        # Files affected (count from tool results)
        signals.files_affected = self._count_files_affected(messages)
        
        # Multi-repo detection
        signals.multi_repo = self._detect_multi_repo(all_text)
        
        # Tool call count
        signals.tool_call_count = self._count_tool_calls(messages)
        
        # Error density
        signals.error_density = self._compute_error_density(messages)
        
        # Correction count (user corrections in conversation)
        signals.correction_count = self._count_corrections(messages)
        
        # Iteration count (from assistant messages)
        signals.iteration_count = self._count_iterations(messages)
        
        # Exit reason
        signals.exit_reason = exit_reason
        
        # Context utilization from usage stats
        if usage_stats:
            signals.context_util_final = self._extract_context_util(usage_stats)
        
        # Detect tests
        signals.has_tests = self._detect_tests(messages)
        
        # Error messages
        signals.error_messages = self._extract_error_messages(messages)
        
        return signals

    def classify(
        self,
        messages: List[Dict[str, Any]],
        signals: TaskSignals,
        exit_reason: str
    ) -> FailureReason:
        """
        Classify the root cause of failure based on signals.
        
        Args:
            messages: Full message history
            signals: Pre-extracted signals
            exit_reason: Why the conversation ended
            
        Returns:
            FailureReason enum value
        """
        # Success path
        if exit_reason in ("completed", "user_interrupted") and signals.correction_count <= 1:
            return FailureReason.SUCCESS
        
        # Iteration exhausted
        if exit_reason in ("iteration_exhausted", "budget_exhausted"):
            if signals.context_util_final > 0.6 and signals.tool_call_count > 10:
                return FailureReason.CONTEXT_OVERFLOW
            return FailureReason.ITERATION_EXHAUSTED
        
        # High error density
        if signals.error_density > 0.3:
            return FailureReason.TOOL_ERROR_STORM
        
        # Trajectory correction loop
        if signals.correction_count >= 3:
            return FailureReason.TRAJECTORY_CORRECTION_LOOP
        
        # Context utilization high
        if signals.context_util_final > 0.75:
            return FailureReason.CONTEXT_DUMMY_ZONE
        
        # Trajectory self-interference (model contradicts itself)
        if self._has_self_contradiction(messages):
            return FailureReason.TRAJECTORY_SELF_INTERFERENCE
        
        # Architecture mismatch (style-sensitive operations)
        if signals.task_type in ("refactor", "architecture") and signals.correction_count >= 2:
            return FailureReason.ARCHITECTURE_MISMATCH
        
        # Missing dependency (import/integration tasks)
        if "import" in signals.task_type or self._has_missing_dependency(messages):
            return FailureReason.MISSING_DEPENDENCY
        
        # Slop detection (high tool use but low effectiveness)
        if signals.tool_call_count > 8 and signals.correction_count == 0 and not signals.has_tests:
            # No tests, high tool use, no corrections = might be slop
            pass  # Can't be certain without user feedback
        
        return FailureReason.UNKNOWN

    # ─── Helper methods ───────────────────────────────────────────────

    def _flatten_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Extract all text content from messages for keyword analysis."""
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        parts.append(part.get("text", ""))
        return " ".join(parts).lower()

    def _classify_task_type(self, text: str) -> str:
        """Classify task type from keywords."""
        scores = {}
        for task_type, keywords in self.TASK_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[task_type] = score
        
        if not scores:
            return "simple"
        
        return max(scores, key=scores.get)

    def _classify_complexity(self, text: str) -> str:
        """Classify complexity from heuristics."""
        if any(kw in text for kw in self.COMPLEXITY_KEYWORDS["high"]):
            return "high"
        if any(kw in text for kw in self.COMPLEXITY_KEYWORDS["medium"]):
            return "medium"
        return "low"

    def _count_files_affected(self, messages: List[Dict[str, Any]]) -> int:
        """Count unique files mentioned in tool calls."""
        files = set()
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                # Match common file paths
                paths = re.findall(r'(?:[\w\-\./]+(?:/[\w\-\.]+)+|\.[a-z]{1,4})', content)
                files.update(p for p in paths if "/" in p or "." in p)
        return len(files)

    def _detect_multi_repo(self, text: str) -> bool:
        """Detect if task spans multiple repositories."""
        repo_indicators = ["frontend/", "backend/", "shared/", "libs/", "packages/"]
        return sum(1 for ind in repo_indicators if ind in text) >= 2

    def _count_tool_calls(self, messages: List[Dict[str, Any]]) -> int:
        """Count total tool calls in conversation."""
        count = 0
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                count += len(msg["tool_calls"])
        return count

    def _compute_error_density(self, messages: List[Dict[str, Any]]) -> float:
        """Compute ratio of tool errors to total tool calls."""
        total_calls = self._count_tool_calls(messages)
        if total_calls == 0:
            return 0.0
        
        errors = 0
        for msg in messages:
            if msg.get("role") == "tool":
                content = str(msg.get("content", "")).lower()
                if any(err in content for err in ["error", "exception", "failed", "traceback"]):
                    errors += 1
        
        return min(1.0, errors / total_calls)

    def _count_corrections(self, messages: List[Dict[str, Any]]) -> int:
        """Count user corrections (corrections imply trajectory issues)."""
        corrections = 0
        for msg in messages:
            if msg.get("role") == "user":
                content = str(msg.get("content", "")).lower()
                # Patterns that indicate user correction
                correction_phrases = [
                    "wrong", "incorrect", "not right", "that's not",
                    "redo", "try again", "don't do that", "stop",
                    "you missed", "you forgot", "you ignored",
                ]
                if any(phrase in content for phrase in correction_phrases):
                    corrections += 1
        return corrections

    def _count_iterations(self, messages: List[Dict[str, Any]]) -> int:
        """Count number of assistant turns (iterations)."""
        return sum(1 for msg in messages if msg.get("role") == "assistant")

    def _extract_context_util(self, usage_stats: Dict[str, Any]) -> float:
        """Extract context utilization from usage stats."""
        # Try prompt_tokens vs context_limit if available
        prompt_tokens = usage_stats.get("prompt_tokens", 0)
        context_limit = usage_stats.get("context_limit", 200000)
        if context_limit > 0:
            return min(1.0, prompt_tokens / context_limit)
        return 0.0

    def _detect_tests(self, messages: List[Dict[str, Any]]) -> bool:
        """Detect if conversation involved test files."""
        for msg in messages:
            content = str(msg.get("content", "")).lower()
            if "test" in content and ("_test." in content or "test/" in content):
                return True
        return False

    def _extract_error_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract error messages from tool results."""
        errors = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                if "error" in content.lower() or "exception" in content.lower():
                    # Grab first line of error
                    first_line = content.split("\n")[0][:200]
                    errors.append(first_line)
        return errors[:5]  # Limit to 5

    def _has_self_contradiction(self, messages: List[Dict[str, Any]]) -> bool:
        """Detect if model contradicted itself across turns."""
        claims = []
        for msg in messages:
            if msg.get("role") == "assistant":
                content = str(msg.get("content", "")).lower()
                # Look for contradicting phrases
                for claim in claims:
                    opposites = [
                        ("i will", "i won't"), ("i did", "i didn't"),
                        ("this is", "this isn't"), ("let's", "let's not"),
                    ]
                    for pos, neg in opposites:
                        if pos in claim and neg in content:
                            return True
                # Store assistant claims for comparison
                if len(content) > 20:
                    claims.append(content[:100])
        return False

    def _has_missing_dependency(self, messages: List[Dict[str, Any]]) -> bool:
        """Detect if conversation had missing dependency issues."""
        for msg in messages:
            content = str(msg.get("content", "")).lower()
            if any(phrase in content for phrase in [
                "module not found", "import error", "no module named",
                "dependency", "not installed", "cannot find module"
            ]):
                return True
        return False
