# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['sklearn.ensemble._forest', 'sklearn.tree._tree', 'sklearn.tree._splitter', 'sklearn.tree._criterion', 'sklearn.tree._utils', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree', 'sklearn.ensemble', 'sklearn.pipeline', 'sklearn.preprocessing', 'sklearn.compose', 'sklearn.metrics', 'sklearn.model_selection', 'sklearn.base', 'sklearn.utils', 'pandas', 'pandas._libs.tslibs.np_datetime', 'pandas._libs.tslibs.nattype', 'numpy', 'numpy.random._pickle', 'joblib', 'scipy', 'scipy.special', 'scipy.special._cdflib', 'scipy.sparse', 'scipy.sparse._matrix', 'scipy.sparse.csgraph', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets']
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('scipy')


block_cipher = None


a = Analysis(
    ['thyroid_cancer_app.py'],
    pathex=[],
    binaries=[],
    datas=[('random_forest_thyroid_cancer_model.pkl', '.')],
    hiddenimports=hiddenimports,
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
    name='Thyroid_Cancer_Predictor',
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
