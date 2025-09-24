#!/usr/bin/env python3
"""Simple build script for Liver Cirrhosis App - for troubleshooting"""

import os
import subprocess
import sys


def simple_build():
    """Simple PyInstaller build without complex options"""

    # Check required files
    required_files = [
        "liver_cirrhosis_app.py",
        "liver_cirrhosis.csv",
        "random_forest_liver_cirrhosis_model.pkl",
        "preprocessor.pkl",
        "scaler_liver_cirrhosis.pkl",
    ]

    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)

    if missing:
        print(f"❌ Missing files: {missing}")
        return False

    print("✅ All required files found")

    # Simple PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",
        "--console",  # Keep console for debugging
        "--name=LiverCirrhosis_Simple",
        f"--add-data=liver_cirrhosis.csv;.",
        f"--add-data=random_forest_liver_cirrhosis_model.pkl;.",
        f"--add-data=best_liver_cirrhosis_model.pkl;.",
        f"--add-data=preprocessor.pkl;.",
        f"--add-data=scaler_liver_cirrhosis.pkl;.",
        f"--add-data=feature_names.pkl;.",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.ensemble",
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtWidgets",
        "liver_cirrhosis_app.py",
    ]

    print("Running simple build...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, timeout=300, capture_output=True, text=True
        )  # 5 min timeout

        if result.returncode == 0:
            print("✅ Build successful!")
            return True
        else:
            print("❌ Build failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Build timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"💥 Error: {e}")
        return False


if __name__ == "__main__":
    print("🏥 Simple Liver Cirrhosis App Builder")
    print("=" * 50)
    success = simple_build()

    if success:
        print("\n🎉 BUILD SUCCESSFUL!")
        print("Find your app: dist/LiverCirrhosis_Simple.exe")
    else:
        print("\n❌ BUILD FAILED")
        sys.exit(1)
