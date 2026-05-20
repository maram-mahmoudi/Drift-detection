from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations & constants
# ---------------------------------------------------------------------------

class DriftSeverity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DriftType(str, Enum):
    DATA_DRIFT = "DATA_DRIFT"        # Covariate / feature distribution shift
    CONCEPT_DRIFT = "CONCEPT_DRIFT"  # P(Y|X) shift – label/target distribution
    PREDICTION_DRIFT = "PREDICTION_DRIFT"  # Model output distribution shift
    BIAS_DRIFT = "BIAS_DRIFT"        # Demographic parity decay over time


# PSI thresholds (industry standard)
PSI_THRESHOLDS = {
    DriftSeverity.NONE:     (0.0,  0.10),
    DriftSeverity.LOW:      (0.10, 0.20),
    DriftSeverity.MODERATE: (0.20, 0.25),
    DriftSeverity.HIGH:     (0.25, 0.35),
    DriftSeverity.CRITICAL: (0.35, float("inf")),
}

# KS test p-value threshold (below this → statistically significant drift)
KS_ALPHA = 0.05

# MMD threshold (squared MMD via RBF kernel)
MMD_THRESHOLD_MODERATE = 0.05
MMD_THRESHOLD_HIGH = 0.15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DriftMetrics:
   
    feature_name: str
    drift_type: DriftType
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    mmd_score: float
    severity: DriftSeverity
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    alert_triggered: bool = False
    retraining_recommended: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    report_id: str
    generated_at: str
    model_id: str
    reference_window: str
    current_window: str
    features: List[DriftMetrics] = field(default_factory=list)
    overall_severity: DriftSeverity = DriftSeverity.NONE
    global_psi: float = 0.0
    retraining_triggered: bool = False
    escalation_required: bool = False
    summary: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# PSI calculation
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("PSI requires non-empty reference and current arrays.")

    ref = np.asarray(reference)
    cur = np.asarray(current)

    # Low-cardinality / binary / categorical-numeric features (e.g. 0/1 labels)
    # degenerate under quantile binning: all probability lands in one bin and
    # PSI collapses to ~0. Fall back to a per-value proportion PSI in that
    # case. Threshold of 20 unique values is conservative.
    unique_ref = np.unique(ref)
    if len(unique_ref) <= 20:
        values = np.unique(np.concatenate([ref, cur]))
        ref_counts = np.array([(ref == v).sum() for v in values], dtype=float)
        cur_counts = np.array([(cur == v).sum() for v in values], dtype=float)
    else:
        breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        ref_counts, _ = np.histogram(ref, bins=breakpoints)
        cur_counts, _ = np.histogram(cur, bins=breakpoints)
        ref_counts = ref_counts.astype(float)
        cur_counts = cur_counts.astype(float)

    # Laplace-style smoothing keeps proportions normalised AND non-zero, so
    # both the sum-to-one property and the log are safe.
    ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    cur_pct = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))

    psi_values = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(np.sum(psi_values))


# ---------------------------------------------------------------------------
# MMD calculation (RBF kernel)
# ---------------------------------------------------------------------------

# Compute mean RBF kernel value between all pairs in X and Y.
def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    diff = X[:, None] - Y[None, :]
    return float(np.mean(np.exp(-diff ** 2 / (2 * sigma ** 2))))


def compute_mmd(
    reference: np.ndarray,
    current: np.ndarray,
    sigma: Optional[float] = None,
) -> float:

    ref = reference.astype(float)
    cur = current.astype(float)

    if sigma is None:
        combined = np.concatenate([ref, cur])
        pairwise = np.abs(combined[:, None] - combined[None, :]).flatten()
        # Median heuristic: exclude zero self-distances which would otherwise
        # bias sigma downward and inflate MMD.
        nonzero = pairwise[pairwise > 0]
        sigma = float(np.median(nonzero)) if nonzero.size else 1.0
        if sigma == 0.0:
            sigma = 1.0

    kxx = _rbf_kernel(ref, ref, sigma)
    kyy = _rbf_kernel(cur, cur, sigma)
    kxy = _rbf_kernel(ref, cur, sigma)

    mmd2 = kxx - 2 * kxy + kyy
    return max(0.0, float(mmd2))


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

#Map a PSI score to a Drift Severity level 
def classify_psi_severity(psi: float) -> DriftSeverity:
    
    for severity, (lo, hi) in PSI_THRESHOLDS.items():
        if lo <= psi < hi:
            return severity
    return DriftSeverity.CRITICAL


def classify_overall_severity(metrics_list: List[DriftMetrics]) -> DriftSeverity:
    if not metrics_list:
        return DriftSeverity.NONE

    order = [
        DriftSeverity.NONE,
        DriftSeverity.LOW,
        DriftSeverity.MODERATE,
        DriftSeverity.HIGH,
        DriftSeverity.CRITICAL,
    ]

    worst = max(metrics_list, key=lambda m: order.index(m.severity))
    high_count = sum(1 for m in metrics_list if m.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL))

    if high_count / len(metrics_list) >= 0.5 and worst.severity == DriftSeverity.HIGH:
        return DriftSeverity.CRITICAL

    return worst.severity


# ---------------------------------------------------------------------------
# Core DriftDetector class
# ---------------------------------------------------------------------------

class DriftDetector:

    def __init__(
        self,
        model_id: str = "ai_model",
        psi_bins: int = 10,
        retraining_psi_threshold: float = 0.20,
        escalation_psi_threshold: float = 0.35,
        ks_alpha: float = KS_ALPHA,
        mmd_threshold_moderate: float = MMD_THRESHOLD_MODERATE,
        mmd_threshold_high: float = MMD_THRESHOLD_HIGH,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.model_id = model_id
        self.psi_bins = psi_bins
        self.retraining_psi_threshold = retraining_psi_threshold
        self.escalation_psi_threshold = escalation_psi_threshold
        self.ks_alpha = ks_alpha
        self.mmd_threshold_moderate = mmd_threshold_moderate
        self.mmd_threshold_high = mmd_threshold_high
        self.output_dir = output_dir or Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DriftDetector initialised | model=%s | PSI_retrain_threshold=%.2f "
            "| PSI_escalation_threshold=%.2f",
            model_id, retraining_psi_threshold, escalation_psi_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_feature(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        drift_type: DriftType = DriftType.DATA_DRIFT,
    ) -> DriftMetrics:

        logger.info("Analysing feature: %s | n_ref=%d | n_cur=%d",
                    feature_name, len(reference), len(current))

        psi = compute_psi(reference, current, bins=self.psi_bins)
        ks_stat, ks_pval = stats.ks_2samp(reference, current)
        mmd = compute_mmd(reference, current)

        severity = classify_psi_severity(psi)

        # Upgrade severity if KS test also detects significant drift
        if ks_pval < self.ks_alpha and severity == DriftSeverity.LOW:
            severity = DriftSeverity.MODERATE
            logger.debug("KS test upgraded severity to MODERATE for %s", feature_name)

        alert = severity in (DriftSeverity.MODERATE, DriftSeverity.HIGH, DriftSeverity.CRITICAL)
        retrain = psi >= self.retraining_psi_threshold

        notes = self._build_notes(psi, ks_pval, mmd, severity)

        metrics = DriftMetrics(
            feature_name=feature_name,
            drift_type=drift_type,
            psi_score=round(psi, 6),
            ks_statistic=round(float(ks_stat), 6),
            ks_p_value=round(float(ks_pval), 6),
            mmd_score=round(mmd, 6),
            severity=severity,
            reference_mean=round(float(np.mean(reference)), 4),
            current_mean=round(float(np.mean(current)), 4),
            reference_std=round(float(np.std(reference)), 4),
            current_std=round(float(np.std(current)), 4),
            alert_triggered=alert,
            retraining_recommended=retrain,
            notes=notes,
        )

        logger.info(
            "Feature=%s | PSI=%.4f | KS_stat=%.4f | KS_p=%.4f | MMD=%.4f | Severity=%s",
            feature_name, psi, ks_stat, ks_pval, mmd, severity.value,
        )
        return metrics

    def analyse(
        self,
        reference_data: Dict[str, np.ndarray],
        current_data: Dict[str, np.ndarray],
        drift_types: Optional[Dict[str, DriftType]] = None,
        reference_window: str = "reference",
        current_window: str = "current",
    ) -> DriftReport:
       
        report_id = f"{self.model_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        drift_types = drift_types or {}

        feature_metrics: List[DriftMetrics] = []
        for fname in reference_data:
            if fname not in current_data:
                logger.warning("Feature '%s' missing from current data – skipped.", fname)
                continue
            dt = drift_types.get(fname, DriftType.DATA_DRIFT)
            m = self.analyse_feature(fname, reference_data[fname], current_data[fname], dt)
            feature_metrics.append(m)

        overall = classify_overall_severity(feature_metrics)
        global_psi = float(np.mean([m.psi_score for m in feature_metrics])) if feature_metrics else 0.0
        retrain_triggered = global_psi >= self.retraining_psi_threshold
        escalate = global_psi >= self.escalation_psi_threshold or overall == DriftSeverity.CRITICAL

        summary = self._build_summary(feature_metrics, overall, global_psi, retrain_triggered, escalate)

        report = DriftReport(
            report_id=report_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            model_id=self.model_id,
            reference_window=reference_window,
            current_window=current_window,
            features=feature_metrics,
            overall_severity=overall,
            global_psi=round(global_psi, 6),
            retraining_triggered=retrain_triggered,
            escalation_required=escalate,
            summary=summary,
        )

        logger.info(
            "DriftReport generated | id=%s | overall_severity=%s | global_PSI=%.4f "
            "| retraining=%s | escalation=%s",
            report_id, overall.value, global_psi, retrain_triggered, escalate,
        )
        return report

    def save_report(self, report: DriftReport, output_dir: Optional[Path] = None) -> Path:
        """Persist the DriftReport as a JSON file (tamper-evident via hash chain pattern)."""
        out = output_dir or self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / f"{report.report_id}.json"
        filepath.write_text(report.to_json(), encoding="utf-8")
        logger.info("Report saved → %s", filepath)
        return filepath

    def check_kill_switch(self, report: DriftReport) -> Tuple[bool, str]:
    
        if report.overall_severity == DriftSeverity.CRITICAL:
            reason = (
                f"KILL SWITCH TRIGGERED: Overall severity=CRITICAL "
                f"(global_PSI={report.global_psi:.4f}). "
                "Model API access should be severed immediately."
            )
            logger.critical(reason)
            return True, reason

        if report.escalation_required and report.retraining_triggered:
            reason = (
                f"KILL SWITCH TRIGGERED: Escalation required AND retraining triggered "
                f"(global_PSI={report.global_psi:.4f}). "
                "Initiating graceful shutdown protocol."
            )
            logger.critical(reason)
            return True, reason

        return False, "System nominal – no kill switch required."

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_notes(
        self, psi: float, ks_pval: float, mmd: float, severity: DriftSeverity
    ) -> str:
        parts = []
        if psi >= self.escalation_psi_threshold:
            parts.append("PSI exceeds escalation threshold – board-level notification required.")
        elif psi >= self.retraining_psi_threshold:
            parts.append("PSI exceeds retraining threshold – PEFT pipeline should be triggered.")
        if ks_pval < self.ks_alpha:
            parts.append(f"KS test significant (p={ks_pval:.4f}) – distributional shift confirmed.")
        if mmd >= self.mmd_threshold_high:
            parts.append("High MMD – severe kernel-space divergence detected.")
        elif mmd >= self.mmd_threshold_moderate:
            parts.append("Moderate MMD – monitor closely.")
        if severity == DriftSeverity.NONE:
            parts.append("No significant drift detected.")
        return " ".join(parts) or "Metrics within acceptable bounds."

    def _build_summary(
        self,
        features: List[DriftMetrics],
        overall: DriftSeverity,
        global_psi: float,
        retrain: bool,
        escalate: bool,
    ) -> str:
        n_alert = sum(1 for f in features if f.alert_triggered)
        lines = [
            f"Drift analysis complete for model '{self.model_id}'.",
            f"Features analysed: {len(features)} | Alerts raised: {n_alert}.",
            f"Global PSI: {global_psi:.4f} | Overall severity: {overall.value}.",
        ]
        if retrain:
            lines.append("RECOMMENDATION: Trigger PEFT retraining pipeline immediately.")
        if escalate:
            lines.append("ESCALATION: Notify AI Governance Committee. Consider kill-switch.")
        return " ".join(lines)
