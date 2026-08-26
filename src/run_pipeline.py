"""Run and validate the reproducible supply-chain control-tower pipeline."""

import subprocess
import sys

STEPS = [
    "src/baseline_forecasting.py",
    "src/sku_level_forecasting.py",
    "src/inventory_optimization.py",
    "src/supplier_risk.py",
    "src/disruption_detection.py",
    "src/create_control_tower.py",
    "src/business_impact.py",
    "src/shap_explainability.py",
    "src/validate_outputs.py",
]


def main():
    for script in STEPS:
        print(f"\n{'=' * 72}\nRUNNING {script}\n{'=' * 72}")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            raise SystemExit(f"Pipeline stopped: {script} exited with code {result.returncode}")
    print("\nPipeline completed and outputs validated successfully.")


if __name__ == "__main__":
    main()
