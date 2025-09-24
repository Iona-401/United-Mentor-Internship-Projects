# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['liver_cirrhosis_app.py'],
    pathex=[],
    binaries=[],
    datas=[('liver_cirrhosis.csv', '.'), ('random_forest_liver_cirrhosis_model.pkl', '.'), ('best_liver_cirrhosis_model.pkl', '.'), ('scaler_liver_cirrhosis.pkl', '.'), ('preprocessor.pkl', '.'), ('feature_names.pkl', '.'), ('shap_explainer.pkl', '.')],
    hiddenimports=['sklearn', 'sklearn.base', 'sklearn.utils', 'sklearn.utils.validation', 'sklearn.utils._param_validation', 'sklearn.exceptions', 'sklearn.ensemble', 'sklearn.ensemble._forest', 'sklearn.ensemble._gb', 'sklearn.ensemble._gradient_boosting', 'sklearn.tree', 'sklearn.tree._tree', 'sklearn.tree._utils', 'sklearn.tree._classes', 'sklearn.tree._criterion', 'sklearn.tree._splitter', 'xgboost', 'xgboost.core', 'xgboost.sklearn', 'xgboost.training', 'xgboost.compat', 'xgboost.libpath', 'xgboost.tracker', 'xgboost.dmatrix', 'xgboost.callback', 'sklearn.preprocessing', 'sklearn.preprocessing._data', 'sklearn.preprocessing._encoders', 'sklearn.preprocessing._label', 'sklearn.compose', 'sklearn.compose._column_transformer', 'sklearn.pipeline', 'sklearn.model_selection', 'sklearn.model_selection._search', 'sklearn.model_selection._split', 'sklearn.model_selection._validation', 'sklearn.metrics', 'sklearn.metrics._classification', 'sklearn.metrics._ranking', 'sklearn.metrics._scorer', 'sklearn.linear_model', 'sklearn.linear_model._base', 'sklearn.linear_model._logistic', 'sklearn.svm', 'sklearn.svm._base', 'sklearn.svm._classes', 'sklearn.svm._libsvm', 'sklearn.neural_network', 'sklearn.neural_network._multilayer_perceptron', 'sklearn.utils', 'sklearn.utils._typedefs', 'sklearn.utils.validation', 'sklearn.utils._array_api', 'sklearn.utils._param_validation', 'sklearn.utils._estimator_html_repr', 'sklearn.utils.multiclass', 'sklearn.base', 'sklearn.neighbors._typedefs', 'sklearn.neighbors._quad_tree', 'joblib', 'pandas', 'pandas.io', 'pandas.io.common', 'pandas.io.parsers', 'numpy', 'numpy.random', 'numpy.random._pickle', 'scipy', 'scipy.special', 'scipy.special._cdflib', 'scipy.special._ufuncs', 'scipy.special._ufuncs_cxx', 'scipy.linalg', 'scipy.linalg._flinalg', 'scipy.sparse', 'scipy.sparse.csgraph', 'scipy.sparse._matrix', 'scipy.sparse._base', 'scipy.optimize', 'scipy.optimize._linesearch', 'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends', 'matplotlib.backends.backend_qt5agg', 'matplotlib.figure', 'shap', 'shap.explainers', 'shap.explainers._tree', 'shap.plots', 'warnings', 'threading', 'concurrent.futures'],
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
    name='Enhanced_Liver_Cirrhosis_Predictor_v2',
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
    icon=['app_icon.ico'],
)
