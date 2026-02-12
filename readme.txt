The model is in hide_and_seek/model.py.
The main function is run_feature_selection_model in hide_and_seek/tools.py

1. Use hide_and_seek/example_experiment.py to run hide_and_seek on your own data.
2. Use hide_and_seek/run_synthetic_tests.py to perform synthetic data experiments.


- SHAP was installed from: https://pypi.org/project/shap/
- xgboost from: https://pypi.org/project/xgboost/
- lasso and random forest from: https://pypi.org/project/scikit-learn/
- code for running, invase, realx, L2X, lime are present in this repo

Environments:
- hide_and_seek, invase, lime are run using the 'hide-and-seek' environment, environment.yml, python 3.9
- realx is run using the 'realx' environment, realx.yml, python 3.10
- l2x is run using the 'l2x2018' environment, environment_l2x2018.yml, python 3.6
- xgboost (and associated SHAP) is run using the 'xgboost' environment, xgboost.yml, python 3.9


