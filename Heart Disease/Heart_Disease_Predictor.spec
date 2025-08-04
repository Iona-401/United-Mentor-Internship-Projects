# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['heart_disease_app.py'],
    pathex=[],
    binaries=[],
    datas=[('heart_disease_model.pkl', '.'), ('heart_disease_scaler.pkl', '.'), ('dataset.csv', '.')],
    hiddenimports=['sklearn.ensemble', 'sklearn.ensemble._forest', 'sklearn.tree', 'sklearn.tree._tree', 'sklearn.preprocessing', 'sklearn.preprocessing._data', 'sklearn.utils._typedefs', 'sklearn.neighbors._typedefs', 'sklearn.neighbors._quad_tree', 'sklearn.tree._utils', 'sklearn.utils.validation', 'sklearn.utils._array_api', 'sklearn.base', 'joblib', 'pandas', 'numpy', 'scipy', 'scipy.special', 'scipy.special._cdflib', 'scipy.special._ufuncs', 'scipy.special._ufuncs_cxx', 'scipy.linalg', 'scipy.sparse', 'scipy.sparse.csgraph', 'scipy.sparse._matrix', 'scipy.sparse._base', 'matplotlib', 'matplotlib.backends.backend_qt5agg', 'seaborn', 'PyQt5'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Heart_Disease_Predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
