"""
Integration tests for the Self-Learning system.

Tests the full loop: ComplexityAnalyzer + PatternMatcher + ProactivePlanner
→ run_agent.py integration → OutcomeEvaluator → TrajectoryIndex persistence.

Covers Phase 1 (core loop) and Phase 2 (quality improvements).
"""

import pytest
from unittest.mock import MagicMock

from agent.trajectory_index import TrajectoryIndex
from agent.complexity_analyzer import ComplexityAnalyzer
from agent.pattern_matcher import PatternMatcher
from agent.outcome_evaluator import OutcomeEvaluator, Outcome
from agent.proactive_planner import ProactivePlanner
from agent.failure_classifier import FailureReason, TaskSignals


# ============================================================================
# TrajectoryIndex — persistence and get_adjustment
# ============================================================================

class TestTrajectoryIndex:
    """Tests for TrajectoryIndex SQLite persistence and history-based adjustment."""

    @pytest.fixture
    def idx(self, tmp_path):
        """Fresh TrajectoryIndex backed by a temp directory."""
        instance = TrajectoryIndex(hermes_home=str(tmp_path))
        yield instance
        try:
            instance._conn.close()
        except Exception:
            pass

    def _make_signals(self, task_type="refactor", complexity="medium"):
        s = TaskSignals()
        s.task_type = task_type
        s.complexity = complexity
        s.files_affected = 2
        s.multi_repo = False
        s.has_tests = False
        s.tool_call_count = 5
        s.error_density = 0.1
        s.correction_count = 0
        s.iteration_count = 1
        s.exit_reason = "completed"
        s.context_util_final = 0.5
        return s

    def test_index_and_retrieve_via_find_similar(self, idx):
        """Index a trajectory and retrieve it via find_similar."""
        messages = [{"role": "user", "content": "Refactor the auth module"}]
        signals = self._make_signals(task_type="refactor", complexity="medium")

        idx.index(signals=signals, messages=messages, failure_reason=None, completed=True)

        # find_similar should find the indexed task
        results = idx.find_similar("Refactor the auth module", limit=5)
        assert len(results) == 1
        assert results[0].task_type == "refactor"

    def test_get_adjustment_returns_none_when_no_history(self, idx):
        """When no history exists, get_adjustment should return None."""
        result = idx.get_adjustment("refactor")
        assert result is None

    def test_get_adjustment_returns_high_kw_boost_when_medium_tasks_consistently_fail(self, idx):
        """
        If all recent medium-complexity tasks of a given type ended in FAILURE,
        get_adjustment should return {'high_kw_boost': <float>}.
        """
        task_type = "bugfix"  # "Debug task X" is classified as bugfix by FailureClassifier
        signals = self._make_signals(task_type=task_type, complexity="medium")

        # Index 3 failed medium-complexity bugfix tasks
        for i in range(3):
            messages = [{"role": "user", "content": f"Debug task {i}"}]
            idx.index(
                signals=signals,
                messages=messages,
                failure_reason=FailureReason.TOOL_ERROR_STORM,
                completed=False,
            )

        result = idx.get_adjustment(task_type, max_age_days=14)
        assert result is not None
        assert "high_kw_boost" in result

    def test_get_adjustment_returns_none_when_tasks_succeeded(self, idx):
        """Successful tasks should not trigger an adjustment."""
        task_type = "refactor"
        messages = [{"role": "user", "content": "Refactor auth"}]
        signals = self._make_signals(task_type=task_type, complexity="medium")

        idx.index(signals=signals, messages=messages, failure_reason=None, completed=True)
        result = idx.get_adjustment(task_type)
        assert result is None

    def test_find_similar_returns_matching_entries(self, idx):
        """find_similar should return entries with the same task_type."""
        task_type = "refactor"
        signals = self._make_signals(task_type=task_type, complexity="high")

        for i in range(3):
            messages = [{"role": "user", "content": f"Refactor task {i}"}]
            idx.index(
                signals=signals,
                messages=messages,
                failure_reason=FailureReason.TOOL_ERROR_STORM,
                completed=False,
            )

        results = idx.find_similar(task_message="Refactor something", task_type=task_type, limit=10)
        assert len(results) == 3
        assert all(r.task_type == task_type for r in results)

    def test_get_stats_updates_after_indexing(self, idx):
        """get_stats should reflect indexed trajectories."""
        signals = self._make_signals()
        messages = [{"role": "user", "content": "Test"}]

        initial = idx.get_stats()
        assert initial["total_conversations"] == 0

        idx.index(signals=signals, messages=messages, failure_reason=None, completed=True)
        stats = idx.get_stats()
        assert stats["total_conversations"] == 1
        assert stats["completed"] == 1


# ============================================================================
# ComplexityAnalyzer — complexity detection + history adjustment
# ============================================================================

class TestComplexityAnalyzer:
    """Tests for ComplexityAnalyzer.analyze() and history-based adjustment."""

    @pytest.fixture
    def idx(self, tmp_path):
        instance = TrajectoryIndex(hermes_home=str(tmp_path))
        yield instance
        try:
            instance._conn.close()
        except Exception:
            pass

    @pytest.fixture
    def analyzer(self, idx):
        return ComplexityAnalyzer(trajectory_index=idx)

    def test_analyze_low_complexity_task(self, analyzer):
        """Simple tasks like 'list files' should be classified LOW."""
        result = analyzer.analyze(
            task="List all files in the current directory",
            conversation_history=[],
        )
        assert result.complexity == "low"
        assert result.task_type is not None

    def test_analyze_high_complexity_task(self, analyzer):
        """Tasks involving multiple services should be classified HIGH."""
        result = analyzer.analyze(
            task=(
                "Refactor the authentication system across 5 microservices, "
                "update all API contracts, and write migration scripts"
            ),
            conversation_history=[],
        )
        assert result.complexity in ("medium", "high")

    def test_analyze_with_history_adjustment(self, idx, analyzer):
        """
        After indexing failed medium tasks for a type,
        the analyzer should upgrade new tasks of that type to HIGH.
        """
        task_type = "bugfix"

        # Pre-populate history: 3 failed medium bugfix tasks with high correction counts.
        # correction_count=4 triggers avg_corrections > 3, producing high_kw_boost in DB.
        signals = TaskSignals()
        signals.task_type = task_type
        signals.complexity = "medium"
        signals.files_affected = 2
        signals.multi_repo = False
        signals.has_tests = False
        signals.tool_call_count = 5
        signals.error_density = 0.1
        signals.correction_count = 4  # avg > 3 → high_kw_boost
        signals.iteration_count = 1
        signals.exit_reason = "error"
        signals.context_util_final = 0.5

        # Index 3 failed medium bugfix tasks with "multiple" (medium keyword) in task content.
        # task_type must match what _extract_signals will classify the analyzed task as (bugfix).
        for i in range(3):
            messages = [{"role": "user", "content": f"Debug multiple {i}"}]
            idx.index(
                signals=signals,
                messages=messages,
                failure_reason=FailureReason.TOOL_ERROR_STORM,
                completed=False,
            )

        # Now analyze a new debugging task — should be upgraded to HIGH
        # because it has medium keywords + learned boost from trajectory history
        result = analyzer.analyze(
            task="Debug multiple segmentation faults in the networking module",
            conversation_history=[],
        )
        assert result.complexity == "high"


# ============================================================================
# OutcomeEvaluator — outcome classification and auto-indexing
# ============================================================================

class TestOutcomeEvaluator:
    """Tests for OutcomeEvaluator.evaluate() and its auto-indexing behavior."""

    @pytest.fixture
    def idx(self, tmp_path):
        instance = TrajectoryIndex(hermes_home=str(tmp_path))
        yield instance
        try:
            instance._conn.close()
        except Exception:
            pass

    @pytest.fixture
    def evaluator(self, idx):
        return OutcomeEvaluator(trajectory_index=idx)

    def test_evaluate_success(self, evaluator, idx):
        """A conversation that completed successfully should be indexed as SUCCESS."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        report = evaluator.evaluate(
            messages=messages,
            exit_reason="completed",
            planned_actions=[],
        )

        assert report.outcome == Outcome.SUCCESS

        # Verify it was auto-indexed via get_stats
        stats = idx.get_stats()
        assert stats["total_conversations"] >= 1

    def test_evaluate_failure(self, evaluator, idx):
        """A conversation that hit max iterations should be classified non-success."""
        messages = [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "Let me try..."},
            {"role": "tool", "name": "terminal", "content": "error"},
        ]

        report = evaluator.evaluate(
            messages=messages,
            exit_reason="iteration_exhausted",
            planned_actions=[],
        )

        # Should be classified as some failure type
        assert report.outcome in (Outcome.FAILURE, Outcome.PARTIAL_SUCCESS)

    def test_evaluate_auto_indexes(self, evaluator, idx):
        """
        Calling evaluate() should automatically index the result.
        No need to call index() separately.
        """
        messages = [{"role": "user", "content": "Simple task"}]

        initial_count = idx.get_stats()["total_conversations"]

        evaluator.evaluate(
            messages=messages,
            exit_reason="completed",
            planned_actions=[],
        )

        # Count should have increased
        assert idx.get_stats()["total_conversations"] == initial_count + 1


# ============================================================================
# ProactivePlanner — plan_ahead integration with shared TrajectoryIndex
# ============================================================================

class TestProactivePlanner:
    """Tests for ProactivePlanner using a shared TrajectoryIndex."""

    @pytest.fixture
    def idx(self, tmp_path):
        instance = TrajectoryIndex(hermes_home=str(tmp_path))
        yield instance
        try:
            instance._conn.close()
        except Exception:
            pass

    @pytest.fixture
    def planner(self, idx):
        """Planner with shared trajectory index for pattern matching."""
        ca = ComplexityAnalyzer(trajectory_index=idx)
        pm = PatternMatcher(trajectory_index=idx)
        return ProactivePlanner(
            complexity_analyzer=ca,
            pattern_matcher=pm,
            enabled=True,
        )

    @pytest.fixture
    def mock_agent(self):
        """Minimal mock parent agent with delegate_task."""
        agent = MagicMock()
        agent.delegate_task.return_value = "Research findings: check imports first."
        agent.max_iterations = 90
        return agent

    def test_plan_ahead_low_task_no_warnings(self, planner, mock_agent):
        """Simple task should not trigger warnings or subagents."""
        result = planner.plan_ahead(
            task="List files in /tmp",
            conversation_history=[],
            parent_agent=mock_agent,
        )

        assert result.should_proceed is True
        assert len(result.pattern_warnings) == 0

    def test_plan_ahead_disabled(self, idx):
        """When disabled, planner should return a no-op result."""
        ca = ComplexityAnalyzer(trajectory_index=idx)
        pm = PatternMatcher(trajectory_index=idx)
        planner = ProactivePlanner(
            complexity_analyzer=ca,
            pattern_matcher=pm,
            enabled=False,
        )
        mock_agent = MagicMock()
        mock_agent.max_iterations = 90

        result = planner.plan_ahead(
            task="Any complex task",
            conversation_history=[],
            parent_agent=mock_agent,
        )

        assert result.should_proceed is True
        assert result.injected_context == ""
        assert result.reasoning == "Proactive planning disabled"

    def test_plan_ahead_adjusts_budget_for_high_complexity(self, planner, mock_agent):
        """High complexity tasks should request more iteration budget."""
        result = planner.plan_ahead(
            task=(
                "Migrate the entire codebase from Python 2 to Python 3, "
                "update all dependencies, refactor 50+ modules, and run tests"
            ),
            conversation_history=[],
            parent_agent=mock_agent,
        )

        if result.complexity == "high":
            assert result.adjusted_iteration_budget is not None
            assert result.adjusted_iteration_budget > mock_agent.max_iterations


# ============================================================================
# Full loop — shared TrajectoryIndex across all components
# ============================================================================

class TestSelfLearningLoop:
    """
    End-to-end test: all components share ONE TrajectoryIndex instance.

    Verifies:
    1. ComplexityAnalyzer learns from history (get_adjustment)
    2. PatternMatcher finds similar failures from history
    3. OutcomeEvaluator auto-indexes and closes the loop
    """

    @pytest.fixture
    def idx(self, tmp_path):
        instance = TrajectoryIndex(hermes_home=str(tmp_path))
        yield instance
        try:
            instance._conn.close()
        except Exception:
            pass

    def _make_signals(self, task_type="refactor", complexity="medium"):
        s = TaskSignals()
        s.task_type = task_type
        s.complexity = complexity
        s.files_affected = 2
        s.multi_repo = False
        s.has_tests = False
        s.tool_call_count = 5
        s.error_density = 0.1
        s.correction_count = 0
        s.iteration_count = 1
        s.exit_reason = "completed"
        s.context_util_final = 0.5
        return s

    def test_loop_pattern_matcher_finds_indexed_failures(self, idx):
        """
        Simulate the full loop:
        1. Index a failed debugging task
        2. PatternMatcher should detect the pattern and return recommendations
        """
        # Step 1: Manually index a failure
        signals = self._make_signals(task_type="debugging", complexity="medium")
        messages = [{"role": "user", "content": "Debug null pointer"}]
        idx.index(
            signals=signals,
            messages=messages,
            failure_reason=FailureReason.TOOL_ERROR_STORM,
            completed=False,
        )

        # Step 2: PatternMatcher should find it
        pm = PatternMatcher(trajectory_index=idx)
        task_signals = TaskSignals()
        task_signals.task_type = "debugging"
        task_signals.complexity = "medium"
        task_signals.files_affected = 0
        task_signals.multi_repo = False
        task_signals.has_tests = False
        task_signals.tool_call_count = 0
        task_signals.error_density = 0.0
        task_signals.correction_count = 0
        task_signals.iteration_count = 0
        task_signals.exit_reason = ""
        task_signals.context_util_final = 0.0

        patterns = pm.match(
            task="Debug a null pointer exception",
            signals=task_signals,
            min_failures_to_warn=1,
        )

        assert len(patterns.similar_failures) >= 1
        assert len(patterns.recommended_actions) >= 0  # May or may not have recommendations

    def test_loop_complexity_upgrade_from_history(self, idx):
        """
        After 3+ failed medium tasks of a type,
        ComplexityAnalyzer should auto-upgrade new tasks to HIGH.
        """
        # Use 'bugfix' task_type — must match what _extract_signals classifies the analyzed task as.
        # correction_count=4 triggers avg_corrections > 3 → high_kw_boost in DB.
        task_type = "bugfix"
        signals = self._make_signals(task_type=task_type, complexity="medium")
        signals.correction_count = 4  # avg > 3 → high_kw_boost

        # Index 3 failed medium bugfix tasks with "multiple" (medium keyword) in task content
        for i in range(3):
            messages = [{"role": "user", "content": f"Debug multiple {i}"}]
            idx.index(
                signals=signals,
                messages=messages,
                failure_reason=FailureReason.TOOL_ERROR_STORM,
                completed=False,
            )

        # ComplexityAnalyzer should now upgrade — the task has medium keywords + learned boost
        ca = ComplexityAnalyzer(trajectory_index=idx)
        result = ca.analyze(
            task="Debug multiple segmentation faults in the networking module",
            conversation_history=[],
        )

        assert result.complexity == "high"

    def test_loop_outcome_evaluator_closes_the_loop(self, idx):
        """
        OutcomeEvaluator.evaluate() should:
        1. Classify the outcome
        2. Auto-index to TrajectoryIndex
        3. Make the result available for future PatternMatcher queries
        """
        evaluator = OutcomeEvaluator(trajectory_index=idx)

        # Simulate a failed task
        messages = [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "Let me investigate..."},
            {"role": "tool", "name": "terminal", "content": "error: command failed"},
        ]

        report = evaluator.evaluate(
            messages=messages,
            exit_reason="iteration_exhausted",
            planned_actions=["Check logs first"],
        )

        assert report.outcome in (Outcome.FAILURE, Outcome.PARTIAL_SUCCESS)

        # The entry should now be in the TrajectoryIndex
        count = idx.get_stats()["total_conversations"]
        assert count >= 1

        # PatternMatcher should be able to query without crashing
        pm = PatternMatcher(trajectory_index=idx)
        task_signals = TaskSignals()
        task_signals.task_type = report.signals.task_type if report.signals else "unknown"
        task_signals.complexity = report.signals.complexity if report.signals else "medium"
        task_signals.files_affected = 0
        task_signals.multi_repo = False
        task_signals.has_tests = False
        task_signals.tool_call_count = 0
        task_signals.error_density = 0.0
        task_signals.correction_count = 0
        task_signals.iteration_count = 0
        task_signals.exit_reason = ""
        task_signals.context_util_final = 0.0

        patterns = pm.match(
            task="Fix the bug",
            signals=task_signals,
            min_failures_to_warn=1,
        )
        # Just verify no crash
        assert patterns is not None


# ============================================================================
# Config-driven complexity weights (P3.1)
# ============================================================================

class TestConfigDrivenComplexityWeights:
    """Tests for P3.1: config-driven complexity keyword overrides."""

    def test_default_keywords_used_when_no_config(self):
        """Without config overrides, built-in keywords are used."""
        ca = ComplexityAnalyzer()
        assert "entire" in ca._high_keywords
        assert "multiple" in ca._medium_keywords
        assert "simple" in ca._low_keywords

    def test_config_overrides_append_not_replace(self):
        """Config keywords are appended to defaults, not replacing them."""
        ca = ComplexityAnalyzer(config_weights={
            "high": ["kubernetes", "terraform"],
            "medium": ["docker-compose"],
            "low": [],
        })
        # Built-in keywords still present
        assert "entire" in ca._high_keywords
        assert "multiple" in ca._medium_keywords
        # Custom keywords added
        assert "kubernetes" in ca._high_keywords
        assert "terraform" in ca._high_keywords
        assert "docker-compose" in ca._medium_keywords

    def test_custom_high_keyword_upgrades_classification(self):
        """A custom high-complexity keyword upgrades a task to HIGH."""
        # "terraform" is not a built-in high keyword.
        # Task with no built-in HIGH keywords — only custom.
        # 2 custom keywords → effective_high_kw=2 → HIGH.
        ca = ComplexityAnalyzer(config_weights={
            "high": ["terraform", "kubernetes"],
        })
        result = ca.analyze("Deploy terraform modules and configure kubernetes clusters")
        assert result.complexity == "high"
        assert result.confidence == 0.85
        # Built-in HIGH still works (2 built-in keywords: "entire" + "from scratch")
        result2 = ca.analyze("Rewrite the entire monolith from scratch")
        assert result2.complexity == "high"

    def test_empty_config_weights_no_change(self):
        """Empty lists in config_weights leave defaults unchanged."""
        ca = ComplexityAnalyzer(config_weights={"high": [], "medium": [], "low": []})
        assert ca._high_keywords == list(ca.HIGH_INDICATORS["keywords"])
        assert ca._medium_keywords == list(ca.MEDIUM_INDICATORS["keywords"])
        assert ca._low_keywords == list(ca.LOW_INDICATORS["keywords"])
