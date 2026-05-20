from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import List

from drift_detector import DriftReport, DriftMetrics, DriftSeverity

logger = logging.getLogger(__name__)

# ANSI colour codes
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _severity_colour(severity: DriftSeverity) -> str:
    mapping = {
        DriftSeverity.NONE:     _GREEN,
        DriftSeverity.LOW:      _CYAN,
        DriftSeverity.MODERATE: _YELLOW,
        DriftSeverity.HIGH:     _RED,
        DriftSeverity.CRITICAL: _RED + _BOLD,
    }
    return mapping.get(severity, "")


def _colour_wrap(text: str, severity: DriftSeverity) -> str:
    if not os.isatty(1):
        return text
    return f"{_severity_colour(severity)}{text}{_RESET}"

# Converts DriftReport → human-readable and machine-readable outputs
class ReportFormatter:

    # Console dashboard
    @staticmethod
    def to_console(report: DriftReport) -> str:
        """Return a formatted string suitable for printing to stdout."""
        lines: List[str] = []
        sep = "=" * 80

        lines.append(sep)
        lines.append(f"  CAIMOM DRIFT DETECTION REPORT   [{report.report_id}]")
        lines.append(sep)
        lines.append(f"  Model        : {report.model_id}")
        lines.append(f"  Generated    : {report.generated_at}")
        lines.append(f"  Ref Window   : {report.reference_window}")
        lines.append(f"  Cur Window   : {report.current_window}")
        lines.append(sep)
        lines.append(f"  {'FEATURE':<28} {'PSI':>7} {'KS_stat':>8} {'KS_p':>8} {'MMD':>8}  SEVERITY")
        lines.append("-" * 80)

        for m in report.features:
            sev_str = _colour_wrap(f"{m.severity.value:<10}", m.severity)
            lines.append(
                f"  {m.feature_name:<28} {m.psi_score:>7.4f} {m.ks_statistic:>8.4f} "
                f"{m.ks_p_value:>8.4f} {m.mmd_score:>8.4f}  {sev_str}"
            )

        lines.append("-" * 80)
        global_sev = _colour_wrap(report.overall_severity.value, report.overall_severity)
        lines.append(f"  GLOBAL PSI   : {report.global_psi:.4f}   OVERALL SEVERITY: {global_sev}")
        lines.append(sep)

        if report.retraining_triggered:
            lines.append(" RETRAINING TRIGGER: PSI threshold exceeded.")
        if report.escalation_required:
            lines.append(" ESCALATION REQUIRED: Notify AI Governance Committee.")
            lines.append("        ")
        lines.append(f"  SUMMARY: {report.summary}")
        lines.append(sep)
        return "\n".join(lines)

    # JSON evidence bundle
    #Write a tamper-evident JSON evidence bundle for audit compliance
    @staticmethod
    def to_json_evidence(report: DriftReport, output_dir: Path) -> Path:
    
        import hashlib, time

        evidence = {
            "schema_version": "1.0",
            "evidence_type": "AI_DRIFT_DETECTION_REPORT",
            "caimom_stage": "Stage 6: Continuous Model Refinement",
            "control_technique": "Drift Detection (PSI + KS + MMD)",
            "report": report.to_dict(),
            "audit_metadata": {
                "generated_epoch": time.time(),
                "generated_utc": report.generated_at,
            },
        }

        raw_json = json.dumps(evidence, indent=2, default=str)
        sha256 = hashlib.sha256(raw_json.encode()).hexdigest()
        evidence["audit_metadata"]["sha256_integrity_hash"] = sha256
        final_json = json.dumps(evidence, indent=2, default=str)

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{report.report_id}_evidence.json"
        path.write_text(final_json, encoding="utf-8")
        logger.info("Evidence bundle saved → %s (SHA256: %s)", path, sha256[:12])
        return path

    # CSV telemetry export
    # Write per-feature metrics to a CSV file for dashboard ingestion
    @staticmethod
    def to_csv(report: DriftReport, output_dir: Path) -> Path:
        
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{report.report_id}_metrics.csv"
        fieldnames = [
            "report_id", "model_id", "generated_at", "feature_name",
            "drift_type", "psi_score", "ks_statistic", "ks_p_value", "mmd_score",
            "severity", "reference_mean", "current_mean",
            "reference_std", "current_std", "alert_triggered", "retraining_recommended",
        ]
        
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in report.features:
                row = {
                    "report_id": report.report_id,
                    "model_id": report.model_id,
                    "generated_at": report.generated_at,
                    "feature_name": m.feature_name,
                    "drift_type": m.drift_type.value,
                    "psi_score": m.psi_score,
                    "ks_statistic": m.ks_statistic,
                    "ks_p_value": m.ks_p_value,
                    "mmd_score": m.mmd_score,
                    "severity": m.severity.value,
                    "reference_mean": m.reference_mean,
                    "current_mean": m.current_mean,
                    "reference_std": m.reference_std,
                    "current_std": m.current_std,
                    "alert_triggered": m.alert_triggered,
                    "retraining_recommended": m.retraining_recommended,
                }
                writer.writerow(row)
        logger.info("CSV metrics saved → %s", path)
        return path
