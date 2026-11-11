"""
End-to-End Master Pipeline Execution Script.
Orchestrates all phases from data generation to model export, evaluation, explainability, and tests.
Enables 100% reproducible execution with a single command:
    python run_pipeline.py
"""

import sys
import time
import subprocess
import os

STEPS = [
    ("Step 1: Synthetic Educational Data Generation", ["python", "-m", "src.data_generator"]),
    ("Step 2: Exploratory Data Analysis & Ethical Audit", ["python", "-m", "src.eda"]),
    ("Step 3: Chronological Splitting & Feature Engineering Pipeline", ["python", "-m", "src.preprocessing"]),
    ("Step 4: Model Training, Comparison & Serialization", ["python", "-m", "src.models"]),
    ("Step 5: Held-out Test Set Evaluation & Error Analysis", ["python", "-m", "src.evaluation"]),
    ("Step 6: SHAP Global Explainability Generation", ["python", "-m", "src.explainability"]),
    ("Step 7: Automated Test Suite Verification", ["python", "-m", "pytest", "tests", "-v"])
]


def run_pipeline():
    print("=" * 80)
    print("🎓 ALGIONS AI LABS - STUDENT PERFORMANCE & AT-RISK ADVISORY PIPELINE")
    print("=" * 80)

    start_total = time.time()
    python_exe = sys.executable

    for title, cmd_args in STEPS:
        cmd_args[0] = python_exe  # Use active python interpreter
        print(f"\n▶ Running {title}...")
        t0 = time.time()
        result = subprocess.run(cmd_args, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"❌ FAILED: {title} (after {elapsed:.2f}s)")
            print("\n--- STDOUT ---")
            print(result.stdout)
            print("\n--- STDERR ---")
            print(result.stderr)
            sys.exit(1)
        else:
            print(f"✅ COMPLETED: {title} in {elapsed:.2f}s")
            if result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                for line in lines[-4:]:
                    print(f"   {line}")

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print("All models, artifacts, figures, and test validations are synchronized.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()

