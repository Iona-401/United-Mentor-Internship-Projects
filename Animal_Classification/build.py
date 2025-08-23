import os
import subprocess
import sys
from pathlib import Path
import shutil


def build_animal_classification_app():
    """Build the dual-stage animal classification app into a standalone executable."""

    # Check if required files exist
    required_files = [
        "animal_classification_app.py",
        "aniClass_EFF_Stage1.pkl",  # Stage 1 model
        "aniClass_EFF_Stage2.pkl",  # Stage 2 model
    ]

    # Optional files that enhance the build
    optional_files = [
        "app_icon.ico",  # Custom icon
        "requirements.txt",  # Dependencies list
    ]

    # Check for missing required files
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Error: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nMake sure you have:")
        print("  - animal_classification_app.py (your dual-stage app)")
        print("  - aniClass_EFF_Stage1.pkl (Stage 1 EfficientNet model)")
        print("  - aniClass_EFF_Stage2.pkl (Stage 2 EfficientNet model)")
        return False

    # Check optional files
    has_icon = os.path.exists("app_icon.ico")

    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    if os.path.exists("build"):
        shutil.rmtree("build")
        print("   Removed build/ directory")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
        print("   Removed dist/ directory")

    # Remove old spec files
    spec_files = [f for f in os.listdir(".") if f.endswith(".spec")]
    for spec_file in spec_files:
        os.remove(spec_file)
        print(f"   Removed {spec_file}")

    print("\n🚀 Building Dual-Stage Animal Classification App...")
    print("📊 Model Architecture: EfficientNetV2B0 (Stage 1 + Stage 2)")
    print("🎯 Target: Windows Executable (.exe)")

    # PyInstaller command configuration
    cmd = [
        "pyinstaller",
        "--onefile",  # Single executable file
        "--windowed",  # No console window
        "--name=Animal_Classification_Dual_Stage",
        "--clean",  # Clean cache
    ]

    # Add icon if available
    if has_icon:
        cmd.extend(["--icon", "app_icon.ico"])
        print("🎨 Using custom icon: app_icon.ico")
    else:
        print(
            "💡 No custom icon found. To add one, place 'app_icon.ico' in this folder."
        )

    # Add model files as data
    data_files = [
        "--add-data=aniClass_EFF_Stage1.pkl;.",  # Stage 1 model
        "--add-data=aniClass_EFF_Stage2.pkl;.",  # Stage 2 model
    ]

    # Check model file sizes
    stage1_size = (
        os.path.getsize("aniClass_EFF_Stage1.pkl") / (1024 * 1024)
        if os.path.exists("aniClass_EFF_Stage1.pkl")
        else 0
    )
    stage2_size = (
        os.path.getsize("aniClass_EFF_Stage2.pkl") / (1024 * 1024)
        if os.path.exists("aniClass_EFF_Stage2.pkl")
        else 0
    )

    print(f"📦 Including model files:")
    print(f"   - Stage 1 Model: {stage1_size:.1f} MB")
    print(f"   - Stage 2 Model: {stage2_size:.1f} MB")
    print(f"   - Total Models: {stage1_size + stage2_size:.1f} MB")

    cmd.extend(data_files)

    # Comprehensive hidden imports for TensorFlow + PyQt5 application
    hidden_imports = [
        # Core TensorFlow
        "--hidden-import=tensorflow",
        "--hidden-import=tensorflow.keras",
        "--hidden-import=tensorflow.keras.models",
        "--hidden-import=tensorflow.keras.layers",
        "--hidden-import=tensorflow.keras.applications",
        "--hidden-import=tensorflow.keras.applications.efficientnet_v2",
        "--hidden-import=tensorflow.keras.preprocessing",
        "--hidden-import=tensorflow.keras.preprocessing.image",
        "--hidden-import=tensorflow.python",
        "--hidden-import=tensorflow.python.saved_model",
        "--hidden-import=keras",
        # Scientific computing
        "--hidden-import=numpy",
        "--hidden-import=scipy",
        "--hidden-import=scipy.special",
        "--hidden-import=scipy.special._cdflib",
        "--hidden-import=scipy.linalg",
        "--hidden-import=scipy.sparse",
        # Image processing
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--hidden-import=PIL.ImageQt",
        # Model serialization
        "--hidden-import=joblib",
        "--hidden-import=pickle",
        # PyQt5 GUI framework
        "--hidden-import=PyQt5",
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtWidgets",
        "--hidden-import=PyQt5.QtGui",
        # Additional TensorFlow dependencies
        "--hidden-import=absl",
        "--hidden-import=absl.logging",
        "--hidden-import=google.protobuf",
        "--hidden-import=h5py",
        "--hidden-import=tensorboard",
        # Math libraries
        "--hidden-import=sklearn.utils._cython_blas",
        "--hidden-import=sklearn.neighbors.typedefs",
        "--hidden-import=sklearn.neighbors.quad_tree",
        "--hidden-import=sklearn.tree._utils",
    ]

    cmd.extend(hidden_imports)

    # Additional PyInstaller options for better compatibility
    additional_options = [
        "--collect-all=tensorflow",  # Collect all TensorFlow modules
        "--collect-submodules=PIL",  # Collect PIL submodules
        "--collect-data=tensorflow",  # Include TensorFlow data files
        "--noupx",  # Don't use UPX compression (better compatibility)
        "--debug=all",  # Debug info (remove for production)
    ]

    cmd.extend(additional_options)

    # Add the main script
    cmd.append("animal_classification_app.py")

    # Execute PyInstaller
    try:
        print("\n⏳ Running PyInstaller...")
        print("   This may take 5-10 minutes due to TensorFlow dependencies...")
        print("   Progress indicators:")

        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Print output in real-time
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if line:
                if "INFO: Loading module" in line:
                    print(f"   📦 {line}")
                elif "WARNING" in line:
                    print(f"   ⚠️  {line}")
                elif "ERROR" in line:
                    print(f"   ❌ {line}")
                elif "Building" in line or "Analyzing" in line:
                    print(f"   🔧 {line}")

        process.wait()

        if process.returncode == 0:
            print("\n✅ Build completed successfully!")

            # Check if executable was created
            exe_path = "dist/Animal_Classification_Dual_Stage.exe"
            if os.path.exists(exe_path):
                exe_size = os.path.getsize(exe_path) / (1024 * 1024)
                print(f"\n📱 Executable Details:")
                print(f"   Location: {exe_path}")
                print(f"   Size: {exe_size:.1f} MB")
                print(f"   Architecture: Dual-Stage EfficientNetV2B0")
                print(f"   Features: Drag & drop, model consolidation")

                # Performance note
                total_size = exe_size
                if total_size > 500:
                    print(f"   📏 Large file size is normal for TensorFlow apps")

                print(f"\n🎯 Ready for distribution!")
                print(
                    f"   The executable includes both AI models and can run standalone"
                )
                print(f"   No additional installation required on target machines")

                return True
            else:
                print("❌ Error: Executable not found in dist/ folder")
                return False
        else:
            print(f"\n❌ Build failed with return code: {process.returncode}")
            print("\n🔍 Troubleshooting steps:")
            print("1. Check if all dependencies are installed:")
            print("   pip install tensorflow PyQt5 Pillow numpy joblib pyinstaller")
            print("2. Try building without --debug=all for cleaner output")
            print("3. Ensure both model files are in the correct directory")
            print("4. Check available disk space (build requires ~2-3 GB temporarily)")
            return False

    except Exception as e:
        print(f"\n❌ Build error: {e}")
        print("\n🔍 Common solutions:")
        print("- Ensure PyInstaller is installed: pip install pyinstaller")
        print("- Check if antivirus is blocking the build process")
        print("- Try running as administrator")
        print("- Ensure sufficient disk space available")
        return False


def check_dependencies():
    """Check if required Python packages are installed."""
    print("🔍 Checking dependencies...")

    # Custom import mapping for packages that don't follow standard naming
    package_imports = {
        "tensorflow": "tensorflow",
        "PyQt5": "PyQt5.QtWidgets",  # Fixed: Import specific module
        "Pillow": "PIL",  # Fixed: Pillow imports as PIL
        "numpy": "numpy",
        "joblib": "joblib",
        "pyinstaller": "PyInstaller",  # Fixed: Capital I
    }

    missing_packages = []

    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"   ❌ {package_name} (missing)")

    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("   All dependencies available! ✅")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("🐾 Animal Classification App Builder")
    print("   Dual-Stage EfficientNetV2B0 Architecture")
    print("=" * 50)

    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before building.")
        sys.exit(1)

    print()  # Blank line

    # Build the application
    success = build_animal_classification_app()

    print("\n" + "=" * 50)
    if success:
        print("🎉 BUILD SUCCESSFUL!")
        print("Your dual-stage animal classification app is ready!")
        print("Find it in: dist/Animal_Classification_Dual_Stage.exe")
        print("\n🚀 Features included:")
        print("   - Dual EfficientNetV2B0 models (Stage 1 + Stage 2)")
        print("   - Intelligent prediction consolidation")
        print("   - Drag & drop image interface")
        print("   - 15 animal classes supported")
        print("   - High accuracy classification")
    else:
        print("❌ BUILD FAILED!")
        print("Check the error messages above and try again.")
        sys.exit(1)
