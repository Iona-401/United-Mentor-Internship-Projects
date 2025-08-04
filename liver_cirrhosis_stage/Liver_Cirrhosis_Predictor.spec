# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['liver_cirrhosis_app.py'],
    pathex=[],
    binaries=[],
    datas=[('random_forest_liver_cirrhosis_model.pkl', '.'), ('liver_cirrhosis.csv', '.')],
    hiddenimports=['sklearn.ensemble', 'sklearn.ensemble._forest', 'sklearn.tree', 'sklearn.tree._tree', 'sklearn.preprocessing', 'sklearn.preprocessing._data', 'sklearn.utils._typedefs', 'sklearn.neighbors._typedefs', 'sklearn.neighbors._quad_tree', 'sklearn.tree._utils', 'sklearn.utils.validation', 'sklearn.utils._array_api', 'sklearn.base', 'sklearn.compose', 'sklearn.compose._column_transformer', 'sklearn.pipeline', 'joblib', 'pandas', 'numpy', 'scipy', 'scipy.special', 'scipy.special._cdflib', 'scipy.special._ufuncs', 'scipy.special._ufuncs_cxx', 'scipy.linalg', 'scipy.sparse', 'scipy.sparse.csgraph', 'scipy.sparse._matrix', 'scipy.sparse._base', 'PyQt5'],
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
    name='Liver_Cirrhosis_Predictor',
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
