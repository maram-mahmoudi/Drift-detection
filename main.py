from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from drift_detector import DriftDetector, DriftType
from synthetic_data import SyntheticDataGenerator, SyntheticDataset
from report_generator import ReportFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("main")

OUTPUT_DIR = Path("outputs")
DATA_DIR   = Path("data")


def run_scenario(name: str, dataset: SyntheticDataset, detector: DriftDetector) -> None:
    print(f"\n{'#'*80}")
    print(f"# SCENARIO: {name.upper()}")
    print(f"# Expected severity: {dataset.expected_severity}")
    print(f"# Description: {dataset.description}")
    print(f"{'#'*80}")

    # Save dataset metadata
    meta_path = DATA_DIR / f"{name}_metadata.json"
    dataset.save_to_json(meta_path)

    # Annotate drift types for special scenarios
    drift_types = {}
    if name == "concept_drift":
        drift_types = {
            "target_label": DriftType.CONCEPT_DRIFT,
            "model_prediction": DriftType.PREDICTION_DRIFT,
        }

    report = detector.analyse(
        reference_data=dataset.reference,
        current_data=dataset.current,
        drift_types=drift_types,
        reference_window="baseline_Q1_2025",
        current_window="production_Q2_2025",
    )

    # Print dashboard
    print(ReportFormatter.to_console(report))

    # Save outputs
    scenario_out = OUTPUT_DIR / name
    detector.save_report(report, scenario_out)
    ReportFormatter.to_json_evidence(report, scenario_out)
    ReportFormatter.to_csv(report, scenario_out)

    # Kill switch check
    triggered, reason = detector.check_kill_switch(report)
    if triggered:
        print(f"\n  *** {reason} ***\n")

    print(f"\n  Outputs saved → {scenario_out}/")


def build_detector(model_id: str = "credit_scorer_v3") -> DriftDetector:
    return DriftDetector(
        model_id=model_id,
        psi_bins=10,
        retraining_psi_threshold=0.20,
        escalation_psi_threshold=0.35,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=" Drift Detection Pipeline (Scripting Modality)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # All scenarios
  python main.py --scenario high_drift    # One scenario
  python main.py --list-scenarios         # List available
        """,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Name of a single scenario to run (default: all).",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenario names and exit.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="credit_scorer_v3",
        help="Model identifier tag embedded in report metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output files.",
    )
    args = parser.parse_args()

    global OUTPUT_DIR, DATA_DIR
    OUTPUT_DIR = Path(args.output_dir)
    DATA_DIR   = Path("data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    gen = SyntheticDataGenerator(n_reference=5000, n_current=2000, seed=42)
    all_scenarios = gen.get_all_scenarios()

    if args.list_scenarios:
        print("Available scenarios:")
        for name, ds in all_scenarios.items():
            print(f"  {name:<20} expected_severity={ds.expected_severity}")
        return

    detector = build_detector(args.model_id)

    if args.scenario:
        if args.scenario not in all_scenarios:
            print(f"ERROR: Unknown scenario '{args.scenario}'. Use --list-scenarios.")
            sys.exit(1)
        run_scenario(args.scenario, all_scenarios[args.scenario], detector)
    else:
        print("Running all scenarios...")
        for name, dataset in all_scenarios.items():
            run_scenario(name, dataset, detector)

    print("\nAll done. Check the 'outputs/' directory for full results.\n")


if __name__ == "__main__":
    main()
