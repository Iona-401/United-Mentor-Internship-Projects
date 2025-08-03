#!/usr/bin/env python3
"""
Build script for Thyroid Cancer App with Random Forest
Clean, optimized approach for Random Forest executable
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=== Thyroid Cancer App Build (Random Forest) ===")
    
    # Check if the model file exists
    model_file = "random_forest_thyroid_cancer_model.pkl"
    if not Path(model_file).exists():
        print(f"❌ Model file '{model_file}' not found!")
        print("Please run 'python main.py' first to generate the model.")
        return False
    
    print(f"✅ Found model file: {model_file}")
    
    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    for folder in ["build", "dist", "__pycache__"]:
        if Path(folder).exists():
            subprocess.run(["rmdir", "/s", "/q", folder], shell=True, check=False)
    
    # Remove spec files
    for spec_file in Path(".").glob("*.spec"):
        spec_file.unlink()
        print(f"🗑️  Removed old spec file: {spec_file}")
    
    # Step 1: Build the executable
    print("🏗️  Building executable with PyInstaller...")
    
    cmd = [
        'pyinstaller',
        '--onefile',                    # Single executable file
        '--windowed',                   # No console window (for GUI)
        '--add-data', f'{model_file};.',  # Include Random Forest model file
        '--hidden-import', 'sklearn.ensemble._forest',   # Random Forest internals
        '--hidden-import', 'sklearn.tree._tree',         # Tree internals
        '--hidden-import', 'sklearn.tree._splitter',     # Tree splitter
        '--hidden-import', 'sklearn.tree._criterion',    # Tree criterion
        '--hidden-import', 'sklearn.tree._utils',        # Tree utils
        '--hidden-import', 'sklearn.utils._cython_blas', # BLAS operations
        '--hidden-import', 'sklearn.neighbors.typedefs', # sklearn typedefs
        '--hidden-import', 'sklearn.neighbors.quad_tree', # quad tree
        '--hidden-import', 'sklearn.tree',               # sklearn tree
        '--hidden-import', 'sklearn.ensemble',           # sklearn ensemble
        '--hidden-import', 'sklearn.pipeline',           # sklearn pipeline
        '--hidden-import', 'sklearn.preprocessing',      # sklearn preprocessing
        '--hidden-import', 'sklearn.compose',            # sklearn compose
        '--hidden-import', 'sklearn.metrics',            # sklearn metrics
        '--hidden-import', 'sklearn.model_selection',    # sklearn model_selection
        '--hidden-import', 'sklearn.base',               # sklearn base
        '--hidden-import', 'sklearn.utils',              # sklearn utils
        '--hidden-import', 'pandas',                     # Include pandas
        '--hidden-import', 'pandas._libs.tslibs.np_datetime', # pandas datetime
        '--hidden-import', 'pandas._libs.tslibs.nattype', # pandas nattype
        '--hidden-import', 'numpy',                      # Include numpy
        '--hidden-import', 'numpy.random._pickle',       # numpy pickle
        '--hidden-import', 'joblib',                     # Include joblib
        '--hidden-import', 'scipy',                      # Include scipy
        '--hidden-import', 'scipy.special',              # Include scipy.special
        '--hidden-import', 'scipy.special._cdflib',      # Include missing scipy module
        '--hidden-import', 'scipy.sparse',               # scipy sparse
        '--hidden-import', 'scipy.sparse._matrix',       # scipy sparse matrix
        '--hidden-import', 'scipy.sparse.csgraph',       # scipy sparse graph
        '--hidden-import', 'PyQt5.QtCore',               # PyQt5 core
        '--hidden-import', 'PyQt5.QtGui',                # PyQt5 gui
        '--hidden-import', 'PyQt5.QtWidgets',            # PyQt5 widgets
        '--collect-submodules', 'sklearn',               # Collect all sklearn submodules
        '--collect-submodules', 'scipy',                 # Collect all scipy submodules
        '--name', 'Thyroid_Cancer_Predictor',            # Output name
        'thyroid_cancer_app.py'                          # Main script
    ]
    
    try:
        print(f"Running PyInstaller...")
        print("This may take a few minutes...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build completed successfully!")
        
        # Check if executable was created
        exe_path = Path("dist/Thyroid_Cancer_Predictor.exe")
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024*1024)
            print(f"✅ Executable created: {exe_path}")
            print(f"📁 Size: {size_mb:.1f} MB")
            print(f"📂 Full path: {exe_path.absolute()}")
            
            # Test if the executable can be launched
            print("\n🧪 Testing executable...")
            test_result = subprocess.run([str(exe_path), "--help"], 
                                       capture_output=True, timeout=10)
            if test_result.returncode == 0:
                print("✅ Executable launches successfully!")
            else:
                print("⚠️  Executable created but may have runtime issues")
                
        else:
            print("❌ Executable was not found in dist/ folder")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print("Error output:")
        print(e.stderr)
        if e.stdout:
            print("Standard output:")
            print(e.stdout)
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Executable test timed out, but build was successful")
    
    print("\n=== Build Complete ===")
    print("📦 Your executable is ready in the 'dist' folder!")
    print("🎯 You can now distribute the .exe file independently.")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
