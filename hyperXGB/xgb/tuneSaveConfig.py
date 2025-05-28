from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
#from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgb.hyperutils import logregobj, accMetric, customgobj
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace.conditions import InCondition
import smac
import os
from smac import Callback
from smac.runhistory import TrialInfo, TrialValue
# for one run, and
#from sklearnex import patch_sklearn
#patch_sklearn()
#from skcuda import svm

# Now scikit-learn will use Intel oneAPI Data Analytics Library (oneDAL) for accelerated computations.

class my_randomForest:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=42)
        # First we create our hyperparameters
        n_estimators = Integer("n_estimators", (5, 30), default=6)# UCI-100
        bootstrap = Categorical("bootstrap", [True, False], default=True)
        min_samples_leaf = Integer("min_samples_leaf", (1, 6), default=5)
        min_samples_split = Integer("min_samples_split", (2, 40), default=3)
        max_depth = Integer("max_depth", (5, 10), default=6)#UCI-40
        #deprecated- max_features = Categorical("max_features", ["auto", "sqrt"], default="auto")

        # Add hyperparameters and conditions to our configspace,eta,  , gamma
        cs.add_hyperparameters([max_depth,
                                n_estimators, bootstrap, min_samples_leaf, min_samples_split])

        return cs

    def init_(self, train_x=None, val_x = None,
              num_class = 0, train_y = None, val_y=None):
        self.train_x = train_x
        self.val_x = val_x
        self.kClasses = num_class
        self.y_val = val_y
        self.y_train = train_y

    def train(self, config: Configuration, seed: int = 42) -> float:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        config_dict = dict(config)
        # custom_results = {}
        rf = RandomForestClassifier(**config_dict, random_state=seed)
        rf.fit(self.train_x, self.y_train)
        predt_c = rf.predict(self.val_x)
        cost = 1 -  metrics.accuracy_score(self.y_val, predt_c)
        return cost

class my_SVM:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=42)
        # First we create our hyperparameters
        # First we create our hyperparameters
        kernel = Categorical("kernel", ["linear", "poly", "rbf", "sigmoid"], default="poly")
        C = Float("C", (0.001, 1000.0), default=1.0, log=True)
        shrinking = Categorical("shrinking", [True, False], default=True)
        degree = Integer("degree", (1, 5), default=3)
        coef = Float("coef0", (0.0, 10.0), default=0.0)
        gamma = Categorical("gamma", ["auto", "scale"], default="auto")
        # gamma_value = Float("gamma_value", (0.0001, 8.0), default=1.0, log=True)

        # Then we create dependencies
        use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
        use_coef = InCondition(child=coef, parent=kernel, values=["poly", "sigmoid"])
        use_gamma = InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"])
        # use_gamma_value = InCondition(child=gamma_value, parent=gamma, values=["value"])

        # Add hyperparameters and conditions to our configspace
        cs.add_hyperparameters([kernel, C, shrinking, degree, coef, gamma]) #, gamma_value
        cs.add_conditions([use_degree, use_coef, use_gamma]) #, use_gamma_value
        return cs

    def init_(self, train_x=None, val_x = None,
              num_class = 0, train_y = None, val_y=None):
        self.train_x = train_x
        self.val_x = val_x
        self.kClasses = num_class
        self.y_val = val_y
        self.y_train = train_y

    def train(self, config: Configuration, seed: int = 42) -> float:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        config_dict = dict(config)
        # if "gamma" in config:
        #     config_dict["gamma"] = config_dict["gamma_value"] if config_dict["gamma"] == "value" else "auto"
        #     config_dict.pop("gamma_value", None)

        classifier = svm.SVC(**config_dict, random_state=seed)
        classifier.fit(self.train_x, self.y_train)
        predt_c = classifier.predict(self.val_x)
        cost = 1 -  metrics.accuracy_score(self.y_val, predt_c)
        return cost


class CustomCallback(Callback):
    def __init__(self, output_path='root/', method=None, name=None, cv=0) -> None:
        self.trials_counter = 0
        self.output_path = output_path
        self.method=method
        self.dataname = name
        self.cv =cv

    def on_start(self, smbo: smac.main.smbo.SMBO) -> None:
        print("Let's start!")
        print("")

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue):
        self.trials_counter += 1
        if self.trials_counter % 2 == 0:
          incumbent = smbo.intensifier.get_incumbent()
          assert incumbent is not None
          res = smbo.runhistory.get_cost(incumbent)
          f = open(os.path.join(self.output_path , ("configs"+self.method+'-'+self.dataname + '_incum-'+str(self.cv) + ".txt")), "a+")

          f.write(str(incumbent ))
          f.write('\n')
          f.close()
          f = open(os.path.join(self.output_path , ("costs"+self.method+'-'+self.dataname + '_incum-'+str(self.cv) + ".txt")), "a+")
          f.write(str(res)+ '\n')
          f.close()
        return None

class tune_MySVM_MyRF():
    # --- here we pass the Euclidean data,
    def __init__(self, method = 'tuneSVM'):
        if method == 'tuneRF':
            self.med = 'tuneRF'
        else:
            self.med = 'tuneSVM'

    def set_data(self, kclass=2, train_x=None, train_y=None, test_x=None, test_y=None,
                 eval_x= None, val_y=None):
        self.num_class = kclass
        self.train_x = train_x
        self.val_x = eval_x
        self.y_train = train_y
        self.y_val = val_y
        self.y_test = test_y
        self.test_x = test_x
    def set_config(self, out_path='', dataname='', cv=0):
        self.output_path = out_path
        self.dataname = dataname
        self.cv = cv

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
        if self.med == 'tuneSVM':
            classifiere = my_SVM()
        else:
            classifiere = my_randomForest()

        classifiere.init_(train_x=self.train_x, val_x=self.val_x,
                          num_class=self.num_class, train_y =self.y_train, val_y=self.y_val)
        scenario = Scenario(
            classifiere.configspace,
            n_trials=15,  # We want to run max 50 trials (combination of config and seed)
        )
        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)
        # Now we use SMAC to find the best hyperparameters
        smac = HyperparameterOptimizationFacade(
            scenario,
            classifiere.train,
            initial_design=initial_design,
            overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
            callbacks=[CustomCallback(output_path=self.output_path, method=self.med, name=self.dataname, cv=self.cv)],
            logging_level=20,
        )
        incumbent = smac.optimize()
        incumbent_cost = smac.validate(incumbent)
        return incumbent_cost, incumbent, smac

    def allocate(self):

        incumbent_cost, incumbent, smac = self.run_method_tune()
        print('finish tuning,--------------------------')
        print(',--------------------------', incumbent_cost)
        if self.med == 'tuneSVM':
            pred = self.predict_SVM_test( incumbent)
        else:
            pred = self.predict_randomforest_test( incumbent)

        return pred




class my_Exgboost:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=42)
        # First we create our hyperparameters
        eta = Float("eta", (0.01, 0.2), default=0.2)
        gamma = Float("gamma", (0.0, 1.0), default=0.0)
        subsample = Float("subsample", (0.5, 1.0), default=1.0)
        colsample_bytree = Float("colsample_bytree", (0.5, 1.0), default=1.0)
        #learning_rate = Categorical("learning_rate", [0.0001, 0.001, 0.01, 0.1], default=0.1)
        colsample_bylevel = Float("colsample_bylevel", (0.5, 1.0), default=1.0)
        colsample_bynode = Float("colsample_bynode", (0.5, 1.0), default=1.0)
        max_depth = Integer("max_depth", (5, 10), default=6)
        n_estimators = Integer("n_estimators", (6, 30), default=6)# uci-100

        # Add hyperparameters and conditions to our configspace,eta, , colsample_bylevel, colsample_bynode , gamma
        cs.add_hyperparameters([max_depth,eta, gamma, n_estimators,
                                subsample, colsample_bytree, colsample_bylevel, colsample_bynode])

        return cs

    def init_(self, train_x=None, val_x = None, test_x = None,
              num_class = 0, val_y=None,test_y = None, Xtrain=None):
        self.dtrain = train_x
        self.dval = val_x
        self.dtest = test_x
        self.kClasses = num_class
        self.y_test = test_y
        self.y_val = val_y
        self.x = Xtrain



    def train(self, config: Configuration, seed: int = 42) -> float:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        config_dict = dict(config)
        num_boost = config['n_estimators']
        config_dict.pop("n_estimators")
        custom_results = {}

        booster_custom = xgb.train({'num_class': self.kClasses,
                                'disable_default_eval_metric': False, **config_dict},
                               self.dtrain,
                               num_boost_round= num_boost,
                               obj=logregobj,
                               #custom_metric=accMetric,
                               evals_result=custom_results,
                               evals=[(self.dval, 'ttest')], verbose_eval=False)#'tree_method': 'gpu_hist',

        # best_iteration = booster_custom.best_iteration
        predt_custom = booster_custom.predict(self.dval) #
        cost = 1 -  metrics.accuracy_score(self.y_val, predt_custom)
        return cost


# for one run, and
class my_Pxgboost:

    def init_(self, train_x=None, val_x = None, test_x = None,
              num_class = 0, val_y=None,test_y = None):
        self.dtrain = train_x
        self.dval = val_x
        self.dtest = test_x
        self.kClasses = num_class
        self.y_test = test_y
        self.y_val = val_y


    def train(self, config: Configuration, seed: int = 42):
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        config_dict = dict(config)
        custom_results = {}

        booster_custom = xgb.train({'num_class': self.kClasses,
                                'disable_default_eval_metric': True, **config_dict},
                               self.dtrain,
                               num_boost_round= 30,
                               obj=customgobj,
                               custom_metric=accMetric,
                               evals_result=custom_results,
                               evals=[(self.dval, 'ttest')], verbose_eval=False)#'tree_method': 'gpu_hist',

        #best_iteration = booster_custom.best_iteration
        predt_custom = booster_custom.predict(self.dval)#, iteration_range=(0, best_iteration)
        cost = 1 -  metrics.accuracy_score(self.y_val, predt_custom)

        return cost, booster_custom


class erade_exgb():
    # --- here we pass the Euclidean data,
    def set_data(self, kclass=2, train_x=None, train_y=None, test_x=None, test_y=None,
                 eval_x= None, val_y=None):
        self.num_class = kclass
        # from sklearn.model_selection import StratifiedKFold
        # skf = StratifiedKFold(n_splits=2)

        # for [train_index, test_index] in skf.split(train_x, train_y):
        #     break
        # # hy_train_x, hy_val_x, hy_train_y, hy_val_y = train_test_split(train_x, train_y, test_size=0.2,
        # #                                                     random_state=42)
        #
        # hy_train_x, hy_val_x, hy_train_y, hy_val_y = train_x[train_index], train_x[test_index], train_y[train_index], \
        # train_y[test_index]
        hy_train_x, hy_val_x, hy_train_y, hy_val_y = train_x, eval_x, train_y, val_y
        self.dtrain = xgb.DMatrix(hy_train_x, hy_train_y)
        self.dval = xgb.DMatrix(hy_val_x, hy_val_y)
        self.y_val = hy_val_y
        self.dtest = xgb.DMatrix(test_x, test_y)
        self.y_test = test_y
        self.x = hy_train_x

    # assign the configure value to Exgboost
    def getConfig(self, config_def: Configuration):
        confini = {}
        confini['subsample'] = config_def['subsample']
        confini['colsample_bylevel'] = config_def['colsample_bylevel']
        confini['colsample_bynode'] = config_def['colsample_bynode']
        confini['colsample_bytree'] = config_def['colsample_bytree']
        # confini['eta'] = config_def['eta']
        # confini['gamma'] = config_def['gamma']
        confini['max_depth'] = config_def['max_depth']
        return confini

    def val(self, config_def):
        config_dict1 = dict(config_def)
        num_boost = config_def['n_estimators']
        config_dict1.pop("n_estimators")
        print(config_dict1)

        custom_results = {}
        booster_custom = xgb.train({'num_class': self.num_class,
                                    'disable_default_eval_metric': True, **config_dict1},
                                   self.dtrain,
                                   num_boost_round=num_boost,
                                   obj=logregobj,

                                   custom_metric=accMetric,
                                   evals_result=custom_results,
                                   evals=[(self.dval, 'ttest')], verbose_eval=False)

        # best_iteration = booster_custom.best_iteration
        predt_custom = booster_custom.predict(self.dtest)  #

        return predt_custom

    def set_config(self, out_path='', dataname='', method= '', cv=0):
        self.output_path = out_path
        self.dataname = dataname
        self.cv = cv
        self.med = method

    def run_exgb(self, trails_num = 200):
        classifiere = my_Exgboost()
        classifiere.init_(train_x=self.dtrain, val_x=self.dval,
                          num_class=self.num_class, val_y=self.y_val, Xtrain=self.x)
        scenario = Scenario(
            classifiere.configspace,
            n_trials=trails_num,  # We want to run max 50 trials (combination of config and seed)
        )
        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)
        # Now we use SMAC to find the best hyperparameters
        smac = HyperparameterOptimizationFacade(
            scenario,
            classifiere.train,
            initial_design=initial_design,
            overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
            callbacks=[CustomCallback(output_path=self.output_path, method=self.med, name=self.dataname, cv=self.cv)],
            logging_level=20,
        )
        incumbent = smac.optimize()
        incumbent_cost = smac.validate(incumbent)
        return incumbent_cost, incumbent, smac

class erade_pxgb():

    def __init__(self):
        self.name = ''
        self._max_depth = [10, 16],  # def-6-6
        self._subsample = [0.5, 0.7],  # range: (0,1] (def 1),, 0.9
        self._eta = [0.05, 0.1],  # 0-1(def 0.3) , 0.3,  0.2

        # --- here we pass the hyperbolic data,

    def set_data(self, kclass=2, train_x=None, train_y=None, test_x=None, test_y=None,
                 pval_x=None, val_y=None):
        self.num_class = kclass
        hy_train_x, hy_val_x, hy_train_y, hy_val_y = train_x, pval_x, train_y, val_y
        self.dtrain = xgb.DMatrix(hy_train_x, hy_train_y)
        self.dval = xgb.DMatrix(hy_val_x, hy_val_y)
        self.y_val = hy_val_y
        self.dtest = xgb.DMatrix(test_x, test_y)
        self.y_test = test_y
        self.x = hy_train_x

    def set_config(self, out_path='', dataname='', method='', cv=0):
        self.output_path = out_path
        self.dataname = dataname
        self.cv = cv
        self.med = method

    def save_incumbent(self, config=None, cost = 1.0):
        f = open(os.path.join(self.output_path,
                              ("configs" + self.med + '-' + self.dataname + '_incum-' + str(self.cv) + ".txt")),
                 "a+")

        f.write(str(config))
        f.write('\n')
        f.close()
        f = open(os.path.join(self.output_path,
                              ("costs" + self.med + '-' + self.dataname + '_incum-' + str(self.cv) + ".txt")), "a+")
        f.write(str(cost) + '\n')
        f.close()

    def run_pxgb(self):

        config_ini = {}
        best_booster = None
        best_cost = 1.0
        for max_depth in self._max_depth[0]:
            config_ini['max_depth'] = max_depth
            for subsample in self._subsample[0]:
                config_ini['subsample'] = subsample
                for eta in self._eta[0]:
                    config_ini['eta'] = eta
                    classifierp = my_Pxgboost()
                    classifierp.init_(train_x=self.dtrain, val_x=self.dval,
                                      num_class=self.num_class, val_y=self.y_val)
                    cost, booster = classifierp.train(config_ini)
                    self.save_incumbent(config_ini, cost)
                    # here we save the booster for later use, otherwise we have to train again.
                    print('pxgb history: ', cost)
                    if cost < best_cost:
                        best_cost = cost
                        best_booster = booster
        return best_cost, best_booster


class classifier_tuneXGB():

    def set_data(self, kclass=2, etrain_x=None, etrain_y=None, etest_x=None, etest_y=None,
                 ptrain_x=None, ptest_x=None, pvalx = None, eval_x= None, val_y=None):
        self.num_class = kclass

        self.pxgb = erade_pxgb()
        self.pxgb.set_data(kclass=kclass, train_x=ptrain_x, train_y=etrain_y, test_x=ptest_x, test_y=etest_y,
                           pval_x=pvalx, val_y=val_y )
        self.exgb = erade_exgb()
        self.exgb.set_data(kclass=kclass, train_x=etrain_x, train_y=etrain_y, test_x=etest_x, test_y=etest_y,
                           eval_x= eval_x, val_y=val_y)

    def set_config(self, out_path='', dataname='', method ='', cv=0):
        self.output_path = out_path
        self.dataname = dataname
        self.cv = cv
        self.med = method

    def allocate(self):

        self.pxgb.set_config(out_path=self.output_path, dataname=self.dataname, method=self.med, cv=self.cv)
        best_cost, pboost = self.pxgb.run_pxgb()
        print('finish pxgboost,--------------------------')
        print('pxgboost,--------------------------', best_cost)
        self.exgb.set_config(out_path=self.output_path, dataname=self.dataname, method=self.med, cv=self.cv)
        incumbent_cost, incumbent, smach = self.exgb.run_exgb(trails_num = 7)
        print('finish exgboost,--------------------------')
        print('exgboost,--------------------------', incumbent_cost)

        if incumbent_cost <= best_cost:
            predt = self.exgb.val(incumbent)
            print('choose exgboost,--------------------------')
        else:
            predt = pboost.predict(self.pxgb.dtest)#, iteration_range=(0, pboost.best_iteration)
            print('choose pxgboost,--------------------------')

        return predt, incumbent, smach


class classifier_tuneSingXGB():

    def set_data(self, kclass=2, etrain_x=None, etrain_y=None, etest_x=None, etest_y=None,
                 ptrain_x=None, ptest_x=None, pvalx = None, eval_x= None, val_y=None):
        self.num_class = kclass

        self.exgb = erade_exgb()
        self.exgb.set_data(kclass=kclass, train_x=etrain_x, train_y=etrain_y, test_x=etest_x, test_y=etest_y,
                           eval_x= eval_x, val_y=val_y)

    def set_config(self, out_path='', dataname='', method='', cv=0):
        self.output_path = out_path
        self.dataname = dataname
        self.cv = cv
        self.med = method

    def allocate(self):
        self.exgb.set_config(out_path=self.output_path, dataname=self.dataname, method= self.med, cv=self.cv)
        incumbent_cost, incumbent, smach = self.exgb.run_exgb(trails_num = 15)
        print('finish exgboost,--------------------------')
        print('exgboost,--------------------------', incumbent_cost)
        predt = self.exgb.val(incumbent)

        return predt, incumbent, smach