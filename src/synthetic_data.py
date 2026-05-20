from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

RNG_SEED = 42


@dataclass
class SyntheticDataset:
    scenario_name: str
    description: str
    reference: Dict[str, np.ndarray]
    current: Dict[str, np.ndarray]
    expected_severity: str
    generation_params: dict

    def save_to_json(self, path: Path) -> None:
        meta = {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "expected_severity": self.expected_severity,
            "generation_params": self.generation_params,
            "reference_shapes": {k: len(v) for k, v in self.reference.items()},
            "current_shapes": {k: len(v) for k, v in self.current.items()},
            "reference_stats": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in self.reference.items()
            },
            "current_stats": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in self.current.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Dataset metadata saved → %s", path)


class SyntheticDataGenerator:

    FEATURE_SPECS = {
        "age":                  dict(mu=42.0,  sigma=12.0,  lo=18,    hi=75),
        "income":               dict(mu=65000, sigma=25000, lo=10000, hi=200000),
        "credit_score":         dict(mu=680.0, sigma=80.0,  lo=300,   hi=850),
        "loan_amount":          dict(mu=85000, sigma=60000, lo=1000,  hi=500000),
        "debt_to_income_ratio": dict(mu=0.35,  sigma=0.15,  lo=0,     hi=1),
    }

    def __init__(self, n_reference: int = 5000, n_current: int = 2000, seed: int = RNG_SEED):
        self.n_reference = n_reference
        self.n_current = n_current
        self.rng = np.random.default_rng(seed)

    # Internal helpers

    def _clipped_normal(self, mu: float, sigma: float, lo: float, hi: float, n: int) -> np.ndarray:
        raw = self.rng.normal(mu, sigma, n * 2)
        clipped = raw[(raw >= lo) & (raw <= hi)]
        if len(clipped) >= n:
            return clipped[:n]
        # top-up if clipping removed too many samples
        return np.clip(self.rng.normal(mu, sigma, n), lo, hi)

    def _build_features(self, specs: Dict[str, dict], n: int) -> Dict[str, np.ndarray]:
        return {
            name: self._clipped_normal(p["mu"], p["sigma"], p["lo"], p["hi"], n)
            for name, p in specs.items()
        }

    def _add_prediction_and_label(
        self,
        features: Dict[str, np.ndarray],
        noise_sigma: float = 0.05,
        positive_rate: float = 0.15,
    ) -> Dict[str, np.ndarray]:
        n = len(next(iter(features.values())))
        # Synthetic logit based on features
        credit_norm = (features["credit_score"] - 300) / 550  # normalise to [0,1]
        dti = features["debt_to_income_ratio"]
        logit = -2 + (-2.5 * credit_norm) + (3.0 * dti) + self.rng.normal(0, noise_sigma, n)
        prob = 1 / (1 + np.exp(-logit))
        prob = np.clip(prob, 0, 1)
        threshold = np.percentile(prob, 100 * (1 - positive_rate))
        label = (prob >= threshold).astype(int)
        return {**features, "model_prediction": prob, "target_label": label.astype(float)}


    # Scenario builders
    def scenario_no_drift(self) -> SyntheticDataset:
        ref = self._build_features(self.FEATURE_SPECS, self.n_reference)
        cur = self._build_features(self.FEATURE_SPECS, self.n_current)
        ref = self._add_prediction_and_label(ref)
        cur = self._add_prediction_and_label(cur)
        return SyntheticDataset(
            scenario_name="NO_DRIFT",
            description="Current distribution identical to reference (same generative parameters).",
            reference=ref,
            current=cur,
            expected_severity="NONE",
            generation_params={"shift_factor": 0.0},
        )

    def scenario_low_drift(self) -> SyntheticDataset:
        ref = self._build_features(self.FEATURE_SPECS, self.n_reference)
        # Slight mean shift: +5% on income, -2% on credit_score
        drifted_specs = dict(self.FEATURE_SPECS)
        drifted_specs["income"] = dict(mu=68250, sigma=26000, lo=10000, hi=200000)
        drifted_specs["credit_score"] = dict(mu=665.0, sigma=80.0, lo=300, hi=850)
        cur = self._build_features(drifted_specs, self.n_current)
        ref = self._add_prediction_and_label(ref)
        cur = self._add_prediction_and_label(cur)
        return SyntheticDataset(
            scenario_name="LOW_DRIFT",
            description="Minor mean shifts in income (+5%) and credit_score (-2%). PSI expected < 0.10.",
            reference=ref,
            current=cur,
            expected_severity="LOW",
            generation_params={"income_shift": "+5%", "credit_score_shift": "-2%"},
        )

    def scenario_moderate_drift(self) -> SyntheticDataset:
        ref = self._build_features(self.FEATURE_SPECS, self.n_reference)
        drifted_specs = dict(self.FEATURE_SPECS)
        drifted_specs["income"] = dict(mu=75000, sigma=30000, lo=10000, hi=200000)
        drifted_specs["credit_score"] = dict(mu=640.0, sigma=90.0, lo=300, hi=850)
        drifted_specs["debt_to_income_ratio"] = dict(mu=0.42, sigma=0.16, lo=0, hi=1)
        cur = self._build_features(drifted_specs, self.n_current)
        ref = self._add_prediction_and_label(ref)
        cur = self._add_prediction_and_label(cur, positive_rate=0.22)
        return SyntheticDataset(
            scenario_name="MODERATE_DRIFT",
            description="Moderate shifts across income, credit_score, and DTI. PSI expected 0.10-0.20.",
            reference=ref,
            current=cur,
            expected_severity="MODERATE",
            generation_params={"income_shift": "+15%", "credit_score_shift": "-6%", "dti_shift": "+20%"},
        )

    def scenario_high_drift(self) -> SyntheticDataset:
        ref = self._build_features(self.FEATURE_SPECS, self.n_reference)
        # Simulate economic recession: income drops, DTI spikes, scores fall
        drifted_specs = dict(self.FEATURE_SPECS)
        drifted_specs["income"] = dict(mu=50000, sigma=20000, lo=10000, hi=200000)
        drifted_specs["credit_score"] = dict(mu=610.0, sigma=95.0, lo=300, hi=850)
        drifted_specs["debt_to_income_ratio"] = dict(mu=0.52, sigma=0.18, lo=0, hi=1)
        drifted_specs["loan_amount"] = dict(mu=70000, sigma=45000, lo=1000, hi=500000)
        cur = self._build_features(drifted_specs, self.n_current)
        ref = self._add_prediction_and_label(ref)
        cur = self._add_prediction_and_label(cur, positive_rate=0.30)
        return SyntheticDataset(
            scenario_name="HIGH_DRIFT",
            description="Recession scenario: income -23%, credit_score -10%, DTI +49%. PSI expected 0.20-0.35.",
            reference=ref,
            current=cur,
            expected_severity="HIGH",
            generation_params={"income_shift": "-23%", "credit_score_shift": "-10%", "dti_shift": "+49%"},
        )

    def scenario_critical_drift(self) -> SyntheticDataset:
        ref = self._build_features(self.FEATURE_SPECS, self.n_reference)
        # Severe distribution change – new customer segment entirely
        drifted_specs = {
            "age":                  dict(mu=28.0,  sigma=5.0,   lo=18,    hi=75),
            "income":               dict(mu=32000, sigma=12000, lo=10000, hi=200000),
            "credit_score":         dict(mu=560.0, sigma=100.0, lo=300,   hi=850),
            "loan_amount":          dict(mu=15000, sigma=8000,  lo=1000,  hi=500000),
            "debt_to_income_ratio": dict(mu=0.65,  sigma=0.18,  lo=0,     hi=1),
        }
        cur = self._build_features(drifted_specs, self.n_current)
        ref = self._add_prediction_and_label(ref)
        cur = self._add_prediction_and_label(cur, positive_rate=0.45)
        return SyntheticDataset(
            scenario_name="CRITICAL_DRIFT",
            description="Completely new customer segment. PSI expected >= 0.35. Kill switch should trigger.",
            reference=ref,
            current=cur,
            expected_severity="CRITICAL",
            generation_params={"scenario": "new_segment_young_subprime"},
        )

    def scenario_concept_drift(self) -> SyntheticDataset:
        ref = self._build_features(self.FEATURE_SPECS, self.n_reference)
        cur = self._build_features(self.FEATURE_SPECS, self.n_current)  # same features
        ref = self._add_prediction_and_label(ref, positive_rate=0.15)
        # Concept drifts: same features but now higher default rate (e.g. pandemic effect)
        cur = self._add_prediction_and_label(cur, positive_rate=0.38, noise_sigma=0.12)
        return SyntheticDataset(
            scenario_name="CONCEPT_DRIFT",
            description="Features stable but target label distribution shifts (default rate: 15%→38%).",
            reference=ref,
            current=cur,
            expected_severity="HIGH",
            generation_params={"default_rate_ref": 0.15, "default_rate_cur": 0.38},
        )

    def get_all_scenarios(self) -> Dict[str, SyntheticDataset]:
        return {
            "no_drift":       self.scenario_no_drift(),
            "low_drift":      self.scenario_low_drift(),
            "moderate_drift": self.scenario_moderate_drift(),
            "high_drift":     self.scenario_high_drift(),
            "critical_drift": self.scenario_critical_drift(),
            "concept_drift":  self.scenario_concept_drift(),
        }
