import os
import subprocess
import sys
from pathlib import Path
import shutil

def build_heart_disease_app():
    """Build the heart disease prediction app into a standalone executable."""
    
    # Check if required files exist
    required_files = [
        "heart_disease_app.py",
        "heart_disease_model.pkl", 
        "heart_disease_scaler.pkl",
        "dataset.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return False
    
    # Clean previous builds
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    
    spec_files = [f for f in os.listdir(".") if f.endswith(".spec")]
    for spec_file in spec_files:
        os.remove(spec_file)
        print(f"Removed {spec_file}")
    
    print("Building Heart Disease Prediction App...")
    print("Using Random Forest model for better compatibility...")
    
    # PyInstaller command with all necessary options
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=Heart_Disease_Predictor",
        "--add-data=heart_disease_model.pkl;.",
        "--add-data=heart_disease_scaler.pkl;.",
        "--add-data=dataset.csv;.",
        "--hidden-import=sklearn.ensemble",
        "--hidden-import=sklearn.ensemble._forest",
        "--hidden-import=sklearn.tree",
        "--hidden-import=sklearn.tree._tree",
        "--hidden-import=sklearn.preprocessing",
        "--hidden-import=sklearn.preprocessing._data",
        "--hidden-import=sklearn.utils._typedefs",
        "--hidden-import=sklearn.neighbors._typedefs",
        "--hidden-import=sklearn.neighbors._quad_tree",
        "--hidden-import=sklearn.tree._utils",
        "--hidden-import=sklearn.utils.validation",
        "--hidden-import=sklearn.utils._array_api",
        "--hidden-import=sklearn.base",
        "--hidden-import=joblib",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=scipy",
        "--hidden-import=scipy.special",
        "--hidden-import=scipy.special._cdflib",
        "--hidden-import=scipy.special._ufuncs",
        "--hidden-import=scipy.special._ufuncs_cxx",
        "--hidden-import=scipy.linalg",
        "--hidden-import=scipy.sparse",
        "--hidden-import=scipy.sparse.csgraph",
        "--hidden-import=scipy.sparse._matrix",
        "--hidden-import=scipy.sparse._base",
        "--hidden-import=matplotlib",
        "--hidden-import=matplotlib.backends.backend_qt5agg",
        "--hidden-import=seaborn",
        "--hidden-import=PyQt5",
        "heart_disease_app.py"
    ]
    
    try:
        print("Running PyInstaller...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Build completed successfully!")
            
            # Check if exe was created
            exe_path = "dist/Heart_Disease_Predictor.exe"
            if os.path.exists(exe_path):
                size_mb = os.path.getsize(exe_path) / (1024 * 1024)
                print(f"Executable created: {exe_path}")
                print(f"Size: {size_mb:.1f} MB")
                return True
            else:
                print("Error: Executable not found in dist folder")
                return False
        else:
            print("Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"Error during build: {e}")
        return False

if __name__ == "__main__":
    success = build_heart_disease_app()
    if success:
        print("\n✅ Heart Disease Prediction App built successfully!")
        print("You can find the executable in the 'dist' folder")
    else:
        print("\n❌ Build failed. Check the error messages above.")
