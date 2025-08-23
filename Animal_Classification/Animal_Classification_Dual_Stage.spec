# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('aniClass_EFF_Stage1.pkl', '.'), ('aniClass_EFF_Stage2.pkl', '.')]
binaries = []
hiddenimports = ['tensorflow', 'tensorflow.keras', 'tensorflow.keras.models', 'tensorflow.keras.layers', 'tensorflow.keras.applications', 'tensorflow.keras.applications.efficientnet_v2', 'tensorflow.keras.preprocessing', 'tensorflow.keras.preprocessing.image', 'tensorflow.python', 'tensorflow.python.saved_model', 'keras', 'numpy', 'scipy', 'scipy.special', 'scipy.special._cdflib', 'scipy.linalg', 'scipy.sparse', 'PIL', 'PIL.Image', 'PIL.ImageQt', 'joblib', 'pickle', 'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui', 'absl', 'absl.logging', 'google.protobuf', 'h5py', 'tensorboard', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils']
datas += collect_data_files('tensorflow')
hiddenimports += collect_submodules('PIL')
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
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('v', None, 'OPTION')],
    name='Animal_Classification_Dual_Stage',
    debug=True,
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
