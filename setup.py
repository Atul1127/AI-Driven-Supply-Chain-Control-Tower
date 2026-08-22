#!/usr/bin/env python3
"""
setup.py
--------
One-click setup script.
Run: python setup.py

Creates virtual environment, installs dependencies, and verifies everything works.
"""

import subprocess
import sys
import os
import platform


def run(cmd, desc=""):
    print(f"\n▶ {desc or cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ❌ FAILED. Check output above.")
        sys.exit(1)
    print(f"  ✅ Done")


def main():
    print("\n" + "="*55)
    print("  Retail Forecasting Project – Setup")
    print("="*55)

    py = sys.executable
    is_win = platform.system() == "Windows"

    # Create venv
    venv_dir = "venv"
    if not os.path.exists(venv_dir):
        run(f"{py} -m venv {venv_dir}", "Creating virtual environment")
    else:
        print(f"\n▶ Virtual environment already exists ({venv_dir}/)")

    # Determine pip path
    if is_win:
        pip = os.path.join(venv_dir, "Scripts", "pip")
        py_venv = os.path.join(venv_dir, "Scripts", "python")
    else:
        pip = os.path.join(venv_dir, "bin", "pip")
        py_venv = os.path.join(venv_dir, "bin", "python")

    run(f"{pip} install --upgrade pip", "Upgrading pip")
    run(f"{pip} install -r requirements.txt", "Installing project dependencies")

    # Verify
    print("\n▶ Verifying installations …")
    checks = ["pandas", "numpy", "matplotlib", "seaborn", "sklearn", "joblib"]
    for lib in checks:
        result = subprocess.run(
            f"{py_venv} -c \"import {lib}; print('{lib}:', {lib}.__version__)\"",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ {result.stdout.strip()}")
        else:
            print(f"  ❌ {lib} – {result.stderr.strip()}")

    print("\n" + "="*55)
    print("  ✅ Setup complete!")
    print()
    if is_win:
        print("  Activate venv:   venv\\Scripts\\activate")
    else:
        print("  Activate venv:   source venv/bin/activate")
    print("  Run project:     python main.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
