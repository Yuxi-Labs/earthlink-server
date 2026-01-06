"""Self-monitoring utilities for agents.

Tracks performance, uncertainty, and diagnostics over a rolling window.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
import math


@dataclass
class UncertaintyMetrics:
    overall: float
    confidence: float
    risk: float
    variability: float


@dataclass
class PerformanceSnapshot:
    timestamp: datetime
    step_duration_ms: float
    decision_confidence: float
    prediction_error: float | None
    status: str
    success: bool
    action_type: str
    goal_progress: float | None = None


@dataclass
class PerformanceReport:
    avg_step_duration_ms: float
    p95_step_duration_ms: float
    avg_decision_confidence: float
    decision_confidence_p10: float
    prediction_accuracy: float
    failure_rate: float
    recent_actions: list[str]
    trend: str
    uncertainty_score: float
    goal_progress: float
    confidence_calibration: float


@dataclass
class Diagnosis:
    probable_cause: str
    severity: str
    recommendations: list[str] = field(default_factory=list)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


class SelfMonitoringModule:
    """Track own performance, uncertainty, and goal progress."""

    def __init__(self, window: int = 100):
        self.window = window
        self.samples: deque[PerformanceSnapshot] = deque(maxlen=window)

    def record_step(
        self,
        step_duration_ms: float,
        decision_confidence: float,
        prediction_error: float | None,
        success: bool,
        status: str,
        action_type: str,
        risk_overall: float | None = None,
        goal_progress: float | None = None,
    ) -> None:
        self.samples.append(
            PerformanceSnapshot(
                timestamp=datetime.now(UTC),
                step_duration_ms=step_duration_ms,
                decision_confidence=max(0.0, min(decision_confidence, 1.0)),
                prediction_error=prediction_error,
                status=status,
                success=success,
                action_type=action_type,
                goal_progress=goal_progress,
            )
        )

    def quantify_uncertainty(
        self,
        decision_confidence: float,
        risk_overall: float | None = None,
        prediction_error: float | None = None,
    ) -> UncertaintyMetrics:
        confidence_component = 1.0 - max(0.0, min(decision_confidence, 1.0))
        risk_component = max(0.0, min(risk_overall, 1.0)) if risk_overall is not None else 0.0
        error_component = max(0.0, min(prediction_error, 1.0)) if prediction_error is not None else 0.0
        variability_component = self._confidence_variability()
        overall = min(1.0, 0.4 * confidence_component + 0.3 * risk_component + 0.2 * error_component + 0.1 * variability_component)
        return UncertaintyMetrics(
            overall=overall,
            confidence=confidence_component,
            risk=risk_component,
            variability=variability_component,
        )

    def _confidence_variability(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        confidences = [s.decision_confidence for s in self.samples]
        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
        return min(1.0, math.sqrt(variance))

    def track_performance(self) -> PerformanceReport:
        durations = [s.step_duration_ms for s in self.samples]
        confidences = [s.decision_confidence for s in self.samples]
        pred_errors = [s.prediction_error for s in self.samples if s.prediction_error is not None]
        failures = [s for s in self.samples if not s.success]
        actions = [s.action_type for s in list(self.samples)[-5:]]
        goal_progress_samples = [s.goal_progress for s in self.samples if s.goal_progress is not None]

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        p95_duration = _percentile(sorted(durations), 0.95) if durations else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        p10_conf = _percentile(sorted(confidences), 0.10) if confidences else 0.0
        pred_accuracy = 1.0 - (sum(pred_errors) / len(pred_errors)) if pred_errors else 1.0
        failure_rate = len(failures) / len(self.samples) if self.samples else 0.0
        goal_progress = sum(goal_progress_samples) / len(goal_progress_samples) if goal_progress_samples else 0.0

        trend = "stable"
        if len(self.samples) >= 6:
            first = sum(durations[:3]) / 3
            last = sum(durations[-3:]) / 3
            if last > first * 1.25:
                trend = "worsening"
            elif last < first * 0.8:
                trend = "improving"

        uncertainty = self.quantify_uncertainty(
            decision_confidence=avg_conf,
            risk_overall=None,
            prediction_error=sum(pred_errors) / len(pred_errors) if pred_errors else None,
        )
        confidence_calibration = abs(pred_accuracy - avg_conf)

        return PerformanceReport(
            avg_step_duration_ms=avg_duration,
            p95_step_duration_ms=p95_duration,
            avg_decision_confidence=avg_conf,
            decision_confidence_p10=p10_conf,
            prediction_accuracy=pred_accuracy,
            failure_rate=failure_rate,
            recent_actions=actions,
            trend=trend,
            uncertainty_score=uncertainty.overall,
            goal_progress=goal_progress,
            confidence_calibration=confidence_calibration,
        )

    def diagnose(self, report: PerformanceReport) -> Diagnosis:
        if report.failure_rate > 0.3:
            return Diagnosis(
                probable_cause="High failure rate",
                severity="high",
                recommendations=[
                    "Reduce action aggressiveness",
                    "Increase validation before acting",
                    "Inspect recent errors",
                ],
            )
        if report.decision_confidence_p10 < 0.35:
            return Diagnosis(
                probable_cause="Low decision confidence",
                severity="medium",
                recommendations=[
                    "Gather more observations before deciding",
                    "Increase exploration to reduce uncertainty",
                ],
            )
        if report.trend == "worsening":
            return Diagnosis(
                probable_cause="Performance degrading",
                severity="medium",
                recommendations=[
                    "Pause to reassess goals",
                    "Refresh world model or retrain policy",
                ],
            )
        return Diagnosis(
            probable_cause="Healthy",
            severity="low",
            recommendations=["Continue current strategy"],
        )

    def get_report(self) -> dict[str, Any]:
        report = self.track_performance()
        diagnosis = self.diagnose(report)
        return {
            "performance": report,
            "diagnosis": diagnosis,
        }
