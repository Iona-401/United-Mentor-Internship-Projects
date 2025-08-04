import os
import subprocess
import sys
from pathlib import Path
import shutil

def build_liver_cirrhosis_app():
    """Build the liver cirrhosis stage prediction app into a standalone executable."""
    
    # Check if required files exist
    required_files = [
        "liver_cirrhosis_app.py",
        "random_forest_liver_cirrhosis_model.pkl", 
        "liver_cirrhosis.csv"
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
    
    print("Building Liver Cirrhosis Stage Prediction App...")
    print("Using optimized Random Forest model (95.5% accuracy)...")
    
    # PyInstaller command with all necessary options
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=Liver_Cirrhosis_Predictor",
        "--add-data=random_forest_liver_cirrhosis_model.pkl;.",
        "--add-data=liver_cirrhosis.csv;.",
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
        "--hidden-import=sklearn.compose",
        "--hidden-import=sklearn.compose._column_transformer",
        "--hidden-import=sklearn.pipeline",
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
        "--hidden-import=PyQt5",
        "liver_cirrhosis_app.py"
    ]
    
    try:
        print("Running PyInstaller...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Build completed successfully!")
            
            # Check if exe was created
            exe_path = "dist/Liver_Cirrhosis_Predictor.exe"
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
    success = build_liver_cirrhosis_app()
    if success:
        print("\n✅ Liver Cirrhosis Stage Prediction App built successfully!")
        print("You can find the executable in the 'dist' folder")
        print("Model accuracy: 95.5% with optimized Random Forest")
    else:
        print("\n❌ Build failed. Check the error messages above.")
