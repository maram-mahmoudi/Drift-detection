# Drift-detection
A scripting-based  AI Assurance and Security Control Technique (AI-ASCT)

## Overview

This project implements a fully deterministic, script-based **AI Model Drift Detection** system aligned with the CAIMOM lifecycle framework. It computes three complementary statistical metrics: **PSI**, **KS**, and **MMD** to detect data drift and concept drift in production AI systems, and automatically triggers governance actions (retraining alerts, escalation, kill switches) based on hard-coded thresholds.

---

## Project Structure

```
drift_detection/
├── main.py                    # CLI entry point
├── requirements.txt
├── README.md
├── src/
│   ├── drift_detector.py      # Core PSI / KS / MMD engine
│   ├── synthetic_data.py      # Synthetic dataset generator (6 scenarios)
│   └── report_generator.py    # Console + JSON + CSV reporters
├── tests/
│   └── test_drift_detector.py # Full unit + integration test suite
├── data/                      # Generated dataset metadata
└── outputs/                   # Reports (JSON evidence + CSV)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run all drift scenarios

```bash
python main.py
```

### 3. Run a single scenario

```bash
python main.py --scenario critical_drift
python main.py --scenario no_drift
python main.py --scenario high_drift
```

### 4. List available scenarios

```bash
python main.py --list-scenarios
```

### 5. Run the test suite

```bash
python -m pytest tests/ -v
```

---

## Scenarios

| Scenario       | Expected Severity | Description                                |
|----------------|-------------------|--------------------------------------------|
| no_drift       | NONE              | Identical distributions                    |
| low_drift      | LOW               | Minor income/credit_score shift            |
| moderate_drift | MODERATE          | Multi-feature moderate shift               |
| high_drift     | HIGH              | Recession: income -23%, credit_score -10%  |
| critical_drift | CRITICAL          | New customer segment — kill switch fires   |
| concept_drift  | HIGH              | Label distribution shift (default 15%→38%) |

---

## Governance Thresholds

| PSI Range   | Severity  | Action                              |
|-------------|-----------|-------------------------------------|
| 0.00 – 0.10 | NONE      | No action                          |
| 0.10 – 0.20 | LOW       | Monitor                            |
| 0.20 – 0.25 | MODERATE  | Alert + schedule PEFT              |
| 0.25 – 0.35 | HIGH      | Alert + trigger retraining          |
| ≥ 0.35      | CRITICAL  | Kill switch + board escalation      |

---

## Outputs

All outputs are written to `outputs/<scenario>/`:
- `<report_id>.json` — Raw drift report
- `<report_id>_evidence.json` — SHA-256 hashed audit evidence bundle
- `<report_id>_metrics.csv` — Per-feature telemetry for dashboards
