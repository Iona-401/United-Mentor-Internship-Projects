# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('aniClass_EFF_Stage1.pkl', '.'), ('aniClass_EFF_Stage2.pkl', '.'), ('aniClass_CNN_enhanced.pkl', '.'), ('class_names.json', '.')]
binaries = []
hiddenimports = ['tensorflow', 'tensorflow.keras', 'tensorflow.keras.models', 'tensorflow.keras.layers', 'tensorflow.keras.applications', 'tensorflow.keras.applications.efficientnet_v2', 'tensorflow.keras.preprocessing', 'tensorflow.keras.preprocessing.image', 'tensorflow.python', 'tensorflow.python.saved_model', 'keras', 'numpy', 'scipy', 'scipy.special', 'scipy.special._cdflib', 'scipy.linalg', 'scipy.sparse', 'PIL', 'PIL.Image', 'cv2', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'joblib', 'pickle', 'absl', 'absl.logging', 'google.protobuf', 'h5py', 'tensorboard', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_tkagg', 'threading', 'json']
datas += collect_data_files('tensorflow')
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('tkinter')
tmp_ret = collect_all('tensorflow')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['animal_classification_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Animal_Classification_Dual_Stage',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon.ico'],
)
