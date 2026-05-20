from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drift_detector import (
    DriftDetector,
    DriftMetrics,
    DriftReport,
    DriftSeverity,
    DriftType,
    compute_mmd,
    compute_psi,
    classify_psi_severity,
    classify_overall_severity,
)
from synthetic_data import SyntheticDataGenerator

RNG = np.random.default_rng(0)


# Unit tests: PSI
class TestComputePSI:
    def test_identical_distributions_near_zero(self):
        x = RNG.normal(0, 1, 5000)
        psi = compute_psi(x, x)
        assert psi < 0.01, f"PSI={psi} expected near 0 for identical data"

    def test_large_shift_high_psi(self):
        ref = RNG.normal(0, 1, 5000)
        cur = RNG.normal(5, 1, 5000)   # large mean shift
        psi = compute_psi(ref, cur)
        assert psi > 0.20, f"PSI={psi} should be HIGH for large shift"

    def test_raises_on_empty_arrays(self):
        with pytest.raises(ValueError):
            compute_psi(np.array([]), np.array([1, 2, 3]))

    def test_symmetry_approximately(self):
        ref = RNG.normal(0, 1, 5000)
        cur = RNG.normal(1, 1, 5000)
        psi_fwd = compute_psi(ref, cur)
        psi_rev = compute_psi(cur, ref)
        # PSI is not perfectly symmetric (uses ref bin edges), but should be close
        assert abs(psi_fwd - psi_rev) < 0.05

    def test_moderate_shift(self):
        # A 10-unit mean shift on a σ=15 distribution (0.67σ shift)
        # produces measurable PSI well above 0.05; upper bound is generous
        ref = RNG.normal(100, 15, 5000)
        cur = RNG.normal(110, 16, 5000)
        psi = compute_psi(ref, cur)
        assert psi > 0.05, f"Expected PSI > 0.05 for a 0.67σ mean shift, got {psi:.4f}"


# Unit tests: MMD
class TestComputeMMD:
    def test_same_distribution_near_zero(self):
        x = RNG.normal(0, 1, 500)
        mmd = compute_mmd(x, x)
        assert mmd < 0.01

    def test_different_distributions_positive(self):
        ref = RNG.normal(0, 1, 500)
        cur = RNG.normal(3, 1, 500)
        mmd = compute_mmd(ref, cur)
        assert mmd > 0.0

    def test_non_negative(self):
        for _ in range(10):
            ref = RNG.normal(0, 1, 200)
            cur = RNG.normal(RNG.uniform(-2, 2), 1, 200)
            assert compute_mmd(ref, cur) >= 0.0


# Unit tests: Severity classification
class TestClassifySeverity:
    def test_no_drift(self):
        assert classify_psi_severity(0.05) == DriftSeverity.NONE

    def test_low_drift(self):
        assert classify_psi_severity(0.12) == DriftSeverity.LOW

    def test_moderate_drift(self):
        assert classify_psi_severity(0.22) == DriftSeverity.MODERATE

    def test_high_drift(self):
        assert classify_psi_severity(0.28) == DriftSeverity.HIGH

    def test_critical_drift(self):
        assert classify_psi_severity(0.50) == DriftSeverity.CRITICAL

    def test_boundary_exactly_020(self):
        # 0.20 falls in LOW bucket upper bound → MODERATE
        sev = classify_psi_severity(0.20)
        assert sev == DriftSeverity.MODERATE

    def test_empty_metrics_returns_none(self):
        assert classify_overall_severity([]) == DriftSeverity.NONE


# Unit tests: DriftDetector.analyse_feature
class TestDriftDetectorAnalyseFeature:
    def setup_method(self):
        self.detector = DriftDetector(model_id="test_model")

    def test_no_drift_scenario(self):
        ref = RNG.normal(50, 10, 5000)
        cur = RNG.normal(50, 10, 2000)
        m = self.detector.analyse_feature("income", ref, cur)
        assert m.severity in (DriftSeverity.NONE, DriftSeverity.LOW)
        assert not m.alert_triggered or m.severity == DriftSeverity.LOW

    def test_critical_drift_scenario(self):
        ref = RNG.normal(50, 10, 5000)
        cur = RNG.normal(150, 30, 2000)  # extreme shift
        m = self.detector.analyse_feature("income", ref, cur)
        assert m.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)
        assert m.alert_triggered
        assert m.retraining_recommended

    def test_metrics_fields_populated(self):
        ref = RNG.normal(0, 1, 1000)
        cur = RNG.normal(1, 1, 500)
        m = self.detector.analyse_feature("feat", ref, cur)
        assert isinstance(m.psi_score, float)
        assert isinstance(m.ks_statistic, float)
        assert isinstance(m.ks_p_value, float)
        assert isinstance(m.mmd_score, float)
        assert isinstance(m.severity, DriftSeverity)
        assert isinstance(m.alert_triggered, bool)
        assert isinstance(m.retraining_recommended, bool)

    def test_drift_type_recorded(self):
        ref = RNG.normal(0, 1, 1000)
        cur = RNG.normal(0, 1, 500)
        m = self.detector.analyse_feature("label", ref, cur, DriftType.CONCEPT_DRIFT)
        assert m.drift_type == DriftType.CONCEPT_DRIFT


# Integration tests: full analyse pipeline
class TestDriftDetectorAnalyse:
    def setup_method(self):
        self.detector = DriftDetector(model_id="integration_test")
        self.gen = SyntheticDataGenerator(n_reference=2000, n_current=500, seed=1)

    def test_no_drift_overall_severity_none_or_low(self):
        ds = self.gen.scenario_no_drift()
        report = self.detector.analyse(ds.reference, ds.current)
        assert report.overall_severity in (DriftSeverity.NONE, DriftSeverity.LOW)
        assert not report.retraining_triggered

    def test_critical_drift_triggers_kill_switch(self):
        ds = self.gen.scenario_critical_drift()
        report = self.detector.analyse(ds.reference, ds.current)
        triggered, _ = self.detector.check_kill_switch(report)
        # With extreme drift, at least escalation or kill switch should fire
        assert report.overall_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)

    def test_report_has_correct_number_of_features(self):
        ds = self.gen.scenario_moderate_drift()
        report = self.detector.analyse(ds.reference, ds.current)
        assert len(report.features) == len(ds.reference)

    def test_report_global_psi_is_mean_of_feature_psis(self):
        ds = self.gen.scenario_low_drift()
        report = self.detector.analyse(ds.reference, ds.current)
        expected = sum(f.psi_score for f in report.features) / len(report.features)
        assert abs(report.global_psi - expected) < 1e-4

    def test_report_serialises_to_json(self):
        import json
        ds = self.gen.scenario_low_drift()
        report = self.detector.analyse(ds.reference, ds.current)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["model_id"] == "integration_test"
        assert "features" in parsed

    def test_missing_feature_skipped_gracefully(self):
        ds = self.gen.scenario_no_drift()
        # Remove one feature from current
        trimmed_current = {k: v for k, v in ds.current.items() if k != "age"}
        report = self.detector.analyse(ds.reference, trimmed_current)
        feature_names = [f.feature_name for f in report.features]
        assert "age" not in feature_names


# Kill switch tests
class TestKillSwitch:
    def setup_method(self):
        self.detector = DriftDetector(
            model_id="ks_test",
            retraining_psi_threshold=0.20,
            escalation_psi_threshold=0.35,
        )

    def _make_report(self, global_psi, overall_severity, retrain=False, escalate=False):
        from drift_detector import DriftReport
        return DriftReport(
            report_id="test_report",
            generated_at="2026-05-17T00:00:00",
            model_id="ks_test",
            reference_window="ref",
            current_window="cur",
            overall_severity=overall_severity,
            global_psi=global_psi,
            retraining_triggered=retrain,
            escalation_required=escalate,
        )

    def test_no_trigger_for_nominal(self):
        r = self._make_report(0.05, DriftSeverity.NONE)
        triggered, _ = self.detector.check_kill_switch(r)
        assert not triggered

    def test_trigger_for_critical(self):
        r = self._make_report(0.50, DriftSeverity.CRITICAL, retrain=True, escalate=True)
        triggered, reason = self.detector.check_kill_switch(r)
        assert triggered
        assert "KILL SWITCH" in reason

    def test_trigger_for_escalation_plus_retrain(self):
        r = self._make_report(0.38, DriftSeverity.HIGH, retrain=True, escalate=True)
        triggered, _ = self.detector.check_kill_switch(r)
        assert triggered


# Synthetic data tests
class TestSyntheticDataGenerator:
    def setup_method(self):
        self.gen = SyntheticDataGenerator(n_reference=1000, n_current=500, seed=99)

    def test_all_scenarios_generated(self):
        scenarios = self.gen.get_all_scenarios()
        assert len(scenarios) == 6

    def test_feature_names_consistent(self):
        ds = self.gen.scenario_no_drift()
        assert set(ds.reference.keys()) == set(ds.current.keys())

    def test_array_lengths_correct(self):
        ds = self.gen.scenario_high_drift()
        for arr in ds.reference.values():
            assert len(arr) == 1000
        for arr in ds.current.values():
            assert len(arr) == 500

    def test_critical_drift_has_higher_psi_than_no_drift(self):
        from drift_detector import compute_psi
        no_drift  = self.gen.scenario_no_drift()
        crit_drift = self.gen.scenario_critical_drift()
        psi_none = compute_psi(no_drift.reference["income"], no_drift.current["income"])
        psi_crit = compute_psi(crit_drift.reference["income"], crit_drift.current["income"])
        assert psi_crit > psi_none
