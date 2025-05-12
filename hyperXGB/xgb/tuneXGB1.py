import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import datasets, svm
from xgb.hyperutils import logregobj, accMetric, customgobj
import itertools


class my_randomForest:
    def __init__(self):
        # Define hyperparameter ranges for grid search
        self.param_grid = {
            "n_estimators": [5, 10, 25, 50, 100],
            "bootstrap": [True, False],
            "min_samples_leaf": [1, 3, 5],
            "min_samples_split": [2, 10, 20, 30],
            "max_depth": [5, 7, 10],
        }

    def init_(
        self, train_x=None, val_x=None, num_class=0, train_y=None, val_y=None
    ):
        self.train_x = train_x
        self.val_x = val_x
        self.kClasses = num_class
        self.y_val = val_y
        self.y_train = train_y

    def grid_search(self, seed=42):
        """Perform a simple grid search on a subset of parameter combinations"""
        best_cost = float("inf")
        best_params = None

        # Generate a subset of all possible parameter combinations to keep runtime reasonable
        param_combinations = []

        # First get all keys and possible values
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        # Sample parameter combinations to avoid exponential explosion
        max_combinations = 20  # Limit the number of combinations to try
        all_combinations = list(itertools.product(*values))

        # Randomly sample combinations if there are too many
        if len(all_combinations) > max_combinations:
            indices = np.random.choice(
                len(all_combinations), max_combinations, replace=False
            )
            sampled_combinations = [all_combinations[i] for i in indices]
        else:
            sampled_combinations = all_combinations

        # Convert to dictionary format
        for combo in sampled_combinations:
            param_dict = {keys[i]: combo[i] for i in range(len(keys))}
            param_combinations.append(param_dict)

        # Try each parameter combination
        for params in param_combinations:
            rf = RandomForestClassifier(**params, random_state=seed)
            rf.fit(self.train_x, self.y_train)
            predt_c = rf.predict(self.val_x)
            cost = 1 - metrics.accuracy_score(self.y_val, predt_c)

            if cost < best_cost:
                best_cost = cost
                best_params = params

        return best_cost, best_params


class my_SVM:
    def __init__(self):
        # Define hyperparameter ranges for grid search
        self.param_grid = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "shrinking": [True, False],
            "degree": [1, 2, 3, 5],  # Only used by poly kernel
            "coef0": [0.0, 1.0, 5.0],  # Only used by poly and sigmoid kernels
            "gamma": ["auto", "scale"],  # For rbf, poly, sigmoid kernels
        }

        # Define parameter dependencies
        self.param_conditions = {
            "degree": {"kernel": ["poly"]},
            "coef0": {"kernel": ["poly", "sigmoid"]},
            "gamma": {"kernel": ["rbf", "poly", "sigmoid"]},
        }

    def init_(
        self, train_x=None, val_x=None, num_class=0, train_y=None, val_y=None
    ):
        self.train_x = train_x
        self.val_x = val_x
        self.kClasses = num_class
        self.y_val = val_y
        self.y_train = train_y

    def is_valid_config(self, config):
        """Check if the parameter configuration respects dependencies"""
        for param, conditions in self.param_conditions.items():
            if param in config:
                for cond_param, valid_values in conditions.items():
                    if (
                        cond_param in config
                        and config[cond_param] not in valid_values
                    ):
                        return False
        return True

    def grid_search(self, seed=42):
        """Perform a simple grid search with parameter dependencies"""
        best_cost = float("inf")
        best_params = None

        # Generate a subset of parameter combinations
        param_combinations = []

        # Try different kernel configurations separately to respect dependencies
        for kernel in self.param_grid["kernel"]:
            # Base configuration for this kernel
            base_config = {"kernel": kernel, "shrinking": True, "C": 1.0}

            # Add kernel-specific parameters
            if kernel == "poly":
                for degree in self.param_grid["degree"]:
                    for coef0 in self.param_grid["coef0"]:
                        for gamma in self.param_grid["gamma"]:
                            config = base_config.copy()
                            config.update(
                                {
                                    "degree": degree,
                                    "coef0": coef0,
                                    "gamma": gamma,
                                }
                            )
                            param_combinations.append(config)

            elif kernel in ["rbf", "sigmoid"]:
                for gamma in self.param_grid["gamma"]:
                    config = base_config.copy()
                    config.update({"gamma": gamma})
                    if kernel == "sigmoid":
                        for coef0 in self.param_grid["coef0"]:
                            config_with_coef = config.copy()
                            config_with_coef.update({"coef0": coef0})
                            param_combinations.append(config_with_coef)
                    else:
                        param_combinations.append(config)

            else:  # linear kernel
                param_combinations.append(base_config)

        # Sample a subset of C values for each configuration
        full_combinations = []
        for config in param_combinations:
            for c in self.param_grid["C"]:
                for shrinking in self.param_grid["shrinking"]:
                    new_config = config.copy()
                    new_config.update({"C": c, "shrinking": shrinking})
                    full_combinations.append(new_config)

        # Randomly sample from full_combinations if there are too many
        max_combinations = 20
        if len(full_combinations) > max_combinations:
            indices = np.random.choice(
                len(full_combinations), max_combinations, replace=False
            )
            sampled_combinations = [full_combinations[i] for i in indices]
        else:
            sampled_combinations = full_combinations

        # Try each parameter combination
        for params in sampled_combinations:
            if self.is_valid_config(params):
                try:
                    classifier = svm.SVC(**params, random_state=seed)
                    classifier.fit(self.train_x, self.y_train)
                    predt_c = classifier.predict(self.val_x)
                    cost = 1 - metrics.accuracy_score(self.y_val, predt_c)

                    if cost < best_cost:
                        best_cost = cost
                        best_params = params
                except Exception as e:
                    print(f"Error with params {params}: {e}")
                    continue

        return best_cost, best_params


class tune_MySVM_MyRF:
    # --- here we pass the Euclidean data,
    def __init__(self, method="tuneSVM"):
        if method == "tuneRF":
            self.med = "tuneRF"
        else:
            self.med = "tuneSVM"

    def set_data(
        self,
        kclass=2,
        train_x=None,
        train_y=None,
        test_x=None,
        test_y=None,
        eval_x=None,
        val_y=None,
    ):
        self.num_class = kclass
        self.train_x = train_x
        self.val_x = eval_x
        self.y_train = train_y
        self.y_val = val_y
        self.y_test = test_y
        self.test_x = test_x

    def predict_SVM_test(self, config_def):
        print(config_def)
        classifier = svm.SVC(**config_def, random_state=42)
        classifier.fit(self.train_x, self.y_train)
        predt_c = classifier.predict(self.test_x)
        return predt_c

    def predict_randomforest_test(self, config_def):
        print(config_def)
        rf = RandomForestClassifier(**config_def, random_state=42)
        rf.fit(self.train_x, self.y_train)
        predt_c = rf.predict(self.test_x)
        return predt_c

    def run_method_tune(self):
        if self.med == "tuneSVM":
            classifier = my_SVM()
        else:
            classifier = my_randomForest()

        classifier.init_(
            train_x=self.train_x,
            val_x=self.val_x,
            num_class=self.num_class,
            train_y=self.y_train,
            val_y=self.y_val,
        )
        incumbent_cost, incumbent = classifier.grid_search()
        return incumbent_cost, incumbent

    def allocate(self):
        incumbent_cost, incumbent = self.run_method_tune()
        print("finish tuning,--------------------------")
        print(",--------------------------", incumbent_cost)
        if self.med == "tuneSVM":
            pred = self.predict_SVM_test(incumbent)
        else:
            pred = self.predict_randomforest_test(incumbent)
        return pred


class my_Exgboost:
    def __init__(self):
        # Define hyperparameter ranges for grid search
        self.param_grid = {
            "eta": [0.01, 0.05, 0.1, 0.2],
            "gamma": [0.0, 0.1, 0.5, 1.0],
            "subsample": [0.5, 0.7, 0.9, 1.0],
            "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
            "colsample_bylevel": [0.4, 0.6, 0.8, 1.0],
            "max_depth": [5, 7, 9],
            "n_estimators": [5, 10, 50, 100],
        }

    def init_(
        self,
        train_x=None,
        val_x=None,
        test_x=None,
        num_class=0,
        val_y=None,
        test_y=None,
        Xtrain=None,
    ):
        self.dtrain = train_x
        self.dval = val_x
        self.dtest = test_x
        self.kClasses = num_class
        self.y_test = test_y
        self.y_val = val_y
        self.x = Xtrain

    def grid_search(self, seed=42):
        """Perform a simple grid search on a subset of parameter combinations"""
        best_cost = float("inf")
        best_params = None

        # Generate a subset of parameter combinations
        param_combinations = []

        # Create a reasonable subset of parameter combinations
        for eta in self.param_grid["eta"]:
            for subsample in [0.7, 1.0]:  # Simplified
                for max_depth in self.param_grid["max_depth"]:
                    for n_estimators in self.param_grid["n_estimators"]:
                        # Fixed values for other parameters to reduce combinations
                        params = {
                            "eta": eta,
                            "gamma": 0.0,
                            "subsample": subsample,
                            "colsample_bytree": 1.0,
                            "colsample_bylevel": 1.0,
                            "max_depth": max_depth,
                            "n_estimators": n_estimators,
                        }
                        param_combinations.append(params)

        # Randomly sample if there are too many combinations
        max_combinations = 20
        if len(param_combinations) > max_combinations:
            indices = np.random.choice(
                len(param_combinations), max_combinations, replace=False
            )
            param_combinations = [param_combinations[i] for i in indices]

        # Try each parameter combination
        for params in param_combinations:
            try:
                num_boost = params["n_estimators"]
                config_dict = params.copy()
                config_dict.pop("n_estimators")

                custom_results = {}
                booster_custom = xgb.train(
                    {
                        "num_class": self.kClasses,
                        "disable_default_eval_metric": True,
                        **config_dict,
                    },
                    self.dtrain,
                    num_boost_round=num_boost,
                    obj=logregobj,
                    custom_metric=accMetric,
                    evals_result=custom_results,
                    evals=[(self.dval, "ttest")],
                    verbose_eval=False,
                )

                predt_custom = booster_custom.predict(self.dval)
                cost = 1 - metrics.accuracy_score(self.y_val, predt_custom)

                if cost < best_cost:
                    best_cost = cost
                    best_params = params
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue

        return best_cost, best_params


class my_Pxgboost:
    def __init__(self):
        # Define fixed hyperparameter combinations to try
        self.param_combinations = [
            {"max_depth": 6, "subsample": 0.5, "eta": 0.1, "gamma": 0.0},
            {"max_depth": 6, "subsample": 0.7, "eta": 0.1, "gamma": 0.0},
            {"max_depth": 16, "subsample": 0.5, "eta": 0.1, "gamma": 0.0},
            {"max_depth": 16, "subsample": 0.7, "eta": 0.1, "gamma": 0.0},
            {"max_depth": 6, "subsample": 0.5, "eta": 0.05, "gamma": 0.0},
            {"max_depth": 6, "subsample": 0.7, "eta": 0.05, "gamma": 0.0},
            {"max_depth": 16, "subsample": 0.5, "eta": 0.05, "gamma": 0.0},
            {"max_depth": 16, "subsample": 0.7, "eta": 0.05, "gamma": 0.0},
        ]

    def init_(
        self,
        train_x=None,
        val_x=None,
        test_x=None,
        num_class=0,
        val_y=None,
        test_y=None,
    ):
        self.dtrain = train_x
        self.dval = val_x
        self.dtest = test_x
        self.kClasses = num_class
        self.y_test = test_y
        self.y_val = val_y

    def grid_search(self):
        """Try all predefined parameter combinations"""
        best_cost = float("inf")
        best_params = None
        best_booster = None

        for params in self.param_combinations:
            try:
                custom_results = {}
                booster_custom = xgb.train(
                    {
                        "num_class": self.kClasses,
                        "disable_default_eval_metric": True,
                        **params,
                    },
                    self.dtrain,
                    num_boost_round=60,
                    obj=customgobj,
                    custom_metric=accMetric,
                    evals_result=custom_results,
                    evals=[(self.dval, "ttest")],
                    verbose_eval=False,
                )

                predt_custom = booster_custom.predict(self.dval)
                cost = 1 - metrics.accuracy_score(self.y_val, predt_custom)

                print("pxgb history: ", cost)
                if cost < best_cost:
                    best_cost = cost
                    best_params = params
                    best_booster = booster_custom
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue

        return best_cost, best_params, best_booster


class erade_exgb:
    # --- here we pass the Euclidean data,
    def set_data(
        self,
        kclass=2,
        train_x=None,
        train_y=None,
        test_x=None,
        test_y=None,
        eval_x=None,
        val_y=None,
    ):
        self.num_class = kclass

        hy_train_x, hy_val_x, hy_train_y, hy_val_y = (
            train_x,
            eval_x,
            train_y,
            val_y,
        )
        self.dtrain = xgb.DMatrix(hy_train_x, hy_train_y)
        self.dval = xgb.DMatrix(hy_val_x, hy_val_y)
        self.y_val = hy_val_y
        self.dtest = xgb.DMatrix(test_x, test_y)
        self.y_test = test_y
        self.x = hy_train_x

    def val(self, config_def):
        config_dict1 = dict(config_def)
        num_boost = config_def["n_estimators"]
        config_dict1.pop("n_estimators")
        print(config_dict1)

        custom_results = {}
        booster_custom = xgb.train(
            {
                "num_class": self.num_class,
                "disable_default_eval_metric": True,
                **config_dict1,
            },
            self.dtrain,
            num_boost_round=num_boost,
            obj=logregobj,
            custom_metric=accMetric,
            evals_result=custom_results,
            evals=[(self.dval, "ttest")],
            verbose_eval=False,
        )

        predt_custom = booster_custom.predict(self.dtest)
        return predt_custom

    def run_exgb(self, trails_num=20):
        classifier = my_Exgboost()
        classifier.init_(
            train_x=self.dtrain,
            val_x=self.dval,
            num_class=self.num_class,
            val_y=self.y_val,
            Xtrain=self.x,
        )
        incumbent_cost, incumbent = classifier.grid_search()
        return incumbent_cost, incumbent


class erade_pxgb:
    def __init__(self):
        self.name = ""
        self.classifier = my_Pxgboost()

    def set_data(
        self,
        kclass=2,
        train_x=None,
        train_y=None,
        test_x=None,
        test_y=None,
        pval_x=None,
        val_y=None,
    ):
        self.num_class = kclass

        hy_train_x, hy_val_x, hy_train_y, hy_val_y = (
            train_x,
            pval_x,
            train_y,
            val_y,
        )
        self.dtrain = xgb.DMatrix(hy_train_x, hy_train_y)
        self.dval = xgb.DMatrix(hy_val_x, hy_val_y)
        self.y_val = hy_val_y
        self.dtest = xgb.DMatrix(test_x, test_y)
        self.y_test = test_y
        self.x = hy_train_x

        self.classifier.init_(
            train_x=self.dtrain,
            val_x=self.dval,
            num_class=self.num_class,
            val_y=self.y_val,
        )

    def run_pxgb(self):
        best_cost, best_params, best_booster = self.classifier.grid_search()
        return best_cost, best_booster


class classifier_tuneXGB:
    def set_data(
        self,
        kclass=2,
        etrain_x=None,
        etrain_y=None,
        etest_x=None,
        etest_y=None,
        ptrain_x=None,
        ptest_x=None,
        pvalx=None,
        eval_x=None,
        val_y=None,
    ):
        self.num_class = kclass

        self.pxgb = erade_pxgb()
        self.pxgb.set_data(
            kclass=kclass,
            train_x=ptrain_x,
            train_y=etrain_y,
            test_x=ptest_x,
            test_y=etest_y,
            pval_x=pvalx,
            val_y=val_y,
        )
        self.exgb = erade_exgb()
        self.exgb.set_data(
            kclass=kclass,
            train_x=etrain_x,
            train_y=etrain_y,
            test_x=etest_x,
            test_y=etest_y,
            eval_x=eval_x,
            val_y=val_y,
        )

    def allocate(self):
        best_cost, pboost = self.pxgb.run_pxgb()
        print("finish pxgboost,--------------------------")
        print("pxgboost,--------------------------", best_cost)

        incumbent_cost, incumbent = self.exgb.run_exgb(trails_num=11)
        print("finish exgboost,--------------------------")
        print("exgboost,--------------------------", incumbent_cost)

        if incumbent_cost <= best_cost:
            predt = self.exgb.val(incumbent)
            print("choose exgboost,--------------------------")
        else:
            predt = pboost.predict(self.pxgb.dtest)
            print("choose pxgboost,--------------------------")

        return predt, incumbent


class classifier_tuneSingXGB:
    def set_data(
        self,
        kclass=2,
        etrain_x=None,
        etrain_y=None,
        etest_x=None,
        etest_y=None,
        ptrain_x=None,
        ptest_x=None,
        pvalx=None,
        eval_x=None,
        val_y=None,
    ):
        self.num_class = kclass

        self.exgb = erade_exgb()
        self.exgb.set_data(
            kclass=kclass,
            train_x=etrain_x,
            train_y=etrain_y,
            test_x=etest_x,
            test_y=etest_y,
            eval_x=eval_x,
            val_y=val_y,
        )

    def allocate(self):
        incumbent_cost, incumbent = self.exgb.run_exgb(trails_num=15)
        print("finish exgboost,--------------------------")
        print("exgboost,--------------------------", incumbent_cost)
        predt = self.exgb.val(incumbent)

        return predt, incumbent
