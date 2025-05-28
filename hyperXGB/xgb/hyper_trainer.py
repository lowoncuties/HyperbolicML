import os
import warnings
from typing import Any
import random
from xgb.poincare import PoincareBall
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import xgboost as xgb
import torch
from ConfigSpace import Configuration
from ignite.utils import setup_logger
from xgb.hyperutils import logregobj, customgobj, accMetric, predict
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from functools import partial


warnings.filterwarnings("ignore")
from .utils import (
    archive_code,
    expanduservars,
    _build_datasets,
    save_results_cc,
    _build_datasets_UCI,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# LOGGER = logging.getLogger(__name__)
from xgb.tuneXGB1 import (
    classifier_tuneXGB,
    tune_MySVM_MyRF,
    classifier_tuneSingXGB,
)

dataname = [
    "abalone",
    "acute-inflammation",
    "acute-nephritis",
    "annealing",
    "audiology-std",
    "balance-scale",
    "balloons",
    "blood",
    "breast-cancer",
    "breast-cancer-wisc",
    "breast-cancer-wisc-diag",
    "breast-cancer-wisc-prog",
    "breast-tissue",
    "car",
    "congressional-voting",
    "conn-bench-sonar-mines-rocks",
    "conn-bench-vowel-deterding",
    "contrac",
    "credit-approval",
    "cylinder-bands",
    "dermatology",
    "echocardiogram",
    "ecoli",
    "energy-y1",
    "energy-y2",
    "fertility",
    "flags",
    "glass",
    "haberman-survival",
    "hayes-roth",
    "heart-cleveland",
    "heart-hungarian",
    "heart-switzerland",
    "heart-va",
    "hepatitis",
    "horse-colic",
    "ilpd-indian-liver",
    "ionosphere",
    "iris",
    "led-display",
    "lenses",
    "libras",
    "lung-cancer",
    "lymphography",
    "mammographic",
    "molec-biol-promoter",
    "monks-1",
    "monks-2",
    "monks-3",
    "oocytes_merluccius_nucleus_4d",
    "oocytes_merluccius_states_2f",
    "oocytes_trisopterus_nucleus_2f",
    "oocytes_trisopterus_states_5b",
    "parkinsons",
    "pima",
    "pittsburg-bridges-MATERIAL",
    "pittsburg-bridges-REL-L",
    "pittsburg-bridges-SPAN",
    "pittsburg-bridges-T-OR-D",
    "pittsburg-bridges-TYPE",
    "planning",
    "post-operative",
    "primary-tumor",
    "seeds",
    "soybean",
    "spect",
    "spectf",
    "statlog-australian-credit",
    "statlog-german-credit",
    "statlog-heart",
    "statlog-image",
    "statlog-vehicle",
    "synthetic-control",
    "teaching",
    "tic-tac-toe",
    "titanic",
    "trains",
    "vertebral-column-2clases",
    "vertebral-column-3clases",
    "wine",
    "wine-quality-red",
    "yeast",
    "zoo",
]


def evaluate_forest(trees, hyp_test_x, labels, params):
    result = Parallel(n_jobs=params["num_jobs"])(
        delayed(partial(tree_predict, hyp_test_x))(trees[i])
        for i in range(params["num_trees"])
    )
    result = np.array(result)

    test_y_preds = result[:, 0]
    test_y_probs = result[:, 1]

    max_probs = np.zeros(len(hyp_test_x))
    test_y_pred = np.zeros(len(hyp_test_x))
    for label in labels:
        mask = (test_y_preds == label).astype(int)
        label_probs = np.sum(test_y_probs * mask, axis=0)

        test_y_pred[label_probs > max_probs] = label
        max_probs[label_probs > max_probs] = label_probs[
            label_probs > max_probs
        ]

    return test_y_pred, label_probs


def tree_predict(hyp_test_x, tree):
    predictions = []
    probabilities = []
    for x in hyp_test_x:
        predicted_class, probability = tree.predict(x)

        predictions.append(predicted_class)
        probabilities.append(probability)

    return predictions, probabilities


def project_weight(w, alpha, ep=1e-5):
    """
    This function can be minimized to find the smallest alpha, which projects
    weights to the closest point so that w * w = -1 (minkowski)

    """
    new_w = w.copy()
    new_w[1:] = (1 + alpha) * new_w[1:]
    new_w[0] = np.sqrt(np.sum((new_w[1:] - ep) ** 2))

    return new_w


# run method for specific data
def run_method(
    params: dict,
    hyp_train_x,
    train_y,
    hyp_test_x,
    test_y,
    labels,
    extrain=None,
    extest=None,
    eval_x=None,
    hval_x=None,
    val_y=None,
):
    #  choose which methods, here 'ExgboostE' or 'ExgboostP' depends use euclidean xgboost on which space data
    if params["method"][:8] == "Pxgboost" or params["method"][:8] == "Exgboost":
        dtrain = xgb.DMatrix(hyp_train_x, train_y)
        dtest = xgb.DMatrix(hyp_test_x, test_y)

        classifier = classifier_hyperxgb()
        classifier.init_(
            kclass=len(labels),
            y_test=test_y,
            X_train=dtrain,
            x_test=dtest,
            mode=params["space"],
            seed=params["seed"],
        )
        config = getConfig(params)
        acc, f1_macro, f1_micro, f1_weight, precision, recall = (
            classifier.train(config, round=params["round"])
        )
    elif params["method"][:12] == "randomforest":
        clf = RandomForestClassifier(
            max_depth=params["max_depth"],
            random_state=0,
            criterion=params["criterion"],
            min_impurity_decrease=params["min_impurity_decrease"],
        )
        clf.fit(hyp_train_x, train_y)
        y_pred = clf.predict(hyp_test_x)
        acc, f1_macro, f1_micro, f1_weight, precision, recall = get_score(
            test_y, y_pred
        )
    elif params["method"][:3] == "SVM":
        clf = SVC(gamma="auto", C=params["c"], kernel=params["kernel"])
        clf.fit(hyp_train_x, train_y)
        y_pred = clf.predict(hyp_test_x)
        acc, f1_macro, f1_micro, f1_weight, precision, recall = get_score(
            test_y, y_pred
        )
    elif params["method"] == "EPxgboost_m":
        classifier = classifier_hyperxgb_EP_m()
        classifier.init_pram(
            kclass=len(labels),
            seed=params["seed"],
            train_x=hyp_train_x,
            train_y=train_y,
            test_x=hyp_test_x,
            test_y=test_y,
            eval_x=eval_x,
            val_y=val_y,
        )

        # classifier.init_pram(kclass=len(labels), seed=params['seed'], train_x=hyp_train_x, train_y=train_y, test_x=hyp_test_x, test_y=test_y)
        config = getConfig(params)
        acc, f1_macro, f1_micro, f1_weight, precision, recall = (
            classifier.train(config, round=params["round"])
        )

    elif params["method"] == "EPxgboost_v":
        classifier = classifier_hyperxgb_EP_v()
        classifier.init_pram(
            kclass=len(labels),
            seed=params["seed"],
            train_x=hyp_train_x,
            train_y=train_y,
            test_x=hyp_test_x,
            test_y=test_y,
            eval_x=eval_x,
            val_y=val_y,
        )
        config = getConfig(params)

        acc, f1_macro, f1_micro, f1_weight, precision, recall = (
            classifier.train(config, round=params["round"])
        )

    elif params["method"] == "tuneEPxgboost":
        classifier1 = classifier_tuneXGB()
        classifier1.set_data(
            kclass=len(labels),
            etrain_x=extrain,
            etrain_y=train_y,
            etest_x=extest,
            etest_y=test_y,
            ptrain_x=hyp_train_x,
            ptest_x=hyp_test_x,
            pvalx=hval_x,
            eval_x=eval_x,
            val_y=val_y,
        )
        y_pred, _, _ = classifier1.allocate()
        acc, f1_macro, f1_micro, f1_weight, precision, recall = get_score(
            test_y, y_pred
        )
        # acc, f1_macro, f1_micro, f1_weight, precision, recall = 0, 0, 0, 0, 0, 0
    elif params["method"] == "tuneSVM" or params["method"] == "tuneRF":
        classifier = tune_MySVM_MyRF(params["method"])
        classifier.set_data(
            kclass=len(labels),
            train_x=extrain,
            train_y=train_y,
            test_x=extest,
            test_y=test_y,
            eval_x=eval_x,
            val_y=val_y,
        )
        y_pred = classifier.allocate()
        acc, f1_macro, f1_micro, f1_weight, precision, recall = get_score(
            test_y, y_pred
        )

    elif params["method"] == "tunesingXGB":
        classifier1 = classifier_tuneSingXGB()
        classifier1.set_data(
            kclass=len(labels),
            etrain_x=extrain,
            etrain_y=train_y,
            etest_x=extest,
            etest_y=test_y,
            ptrain_x=hyp_train_x,
            ptest_x=hyp_test_x,
            pvalx=hval_x,
            eval_x=eval_x,
            val_y=val_y,
        )
        y_pred, _, _ = classifier1.allocate()
        acc, f1_macro, f1_micro, f1_weight, precision, recall = get_score(
            test_y, y_pred
        )

    else:
        acc, f1_macro, f1_micro, f1_weight, precision, recall = 0, 0, 0, 0, 0, 0
        print("there is no such method")
    return acc, f1_macro, f1_micro, f1_weight, precision, recall


# for gaussian data and wordnet data
def run_train(params: dict, params_file):
    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(
        (params["output_path"] + "//" + params["method"])
    )
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logger(
        name=params["dataset_file"],
        format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s",
        reset=False,
        filepath=os.path.join(
            params["output_path"],
            (params["dataset_file"] + params["method"] + ".txt"),
        ),
    )

    archive_code(output_path, params_file)

    logger.info("%d GPUs available", torch.cuda.device_count())
    logger.info(
        "Using seed %d on class %d", params["seed"], params["class_label"]
    )

    for gassid in range(1, (params["data_num"] + 1)):
        params["class_label"] = gassid
        full_train_x, full_train_y, full_test_x, full_test_y, space = (
            _build_datasets(params)
        )

        hyp_f1_micro = []
        hyp_f1_macro = []
        hyp_f1_weight = []
        hyp_acc = []
        hyp_recall = []
        hyp_precision = []

        labels = np.unique(np.concatenate((full_train_y, full_test_y)))
        multi_class = True if len(labels) > 2 else False
        params["multi_class"] = multi_class
        for seedid in range(params["folds"]):
            train_x, test_x, train_y, test_y = train_test_split(
                full_train_x,
                full_train_y,
                test_size=0.3,
                random_state=(params["seed"] + seedid),
            )
            logger.info(
                "%d train and %d test samples for fold %d",
                len(train_x),
                len(test_x),
                seedid,
            )
            if space == "euclidean":
                logger.info("Mapping to hyperbolic space")
                caredisk = PoincareBall(1)
                hyp_train_x = caredisk.expmap0(train_x, 1.0)
                hyp_test_x = caredisk.expmap0(test_x, 1.0)

            else:
                hyp_train_x = train_x
                hyp_test_x = test_x

            acc, f1_macro, f1_micro, f1_weight, precesion, recall = run_method(
                params, hyp_train_x, train_y, hyp_test_x, test_y, labels
            )

            logger.info(
                "Hyperbolic tree f1 micro: %.4f, f1 macro: %.4f, f1_weight: %.4f. acc %.2f",
                f1_micro,
                f1_macro,
                f1_weight,
                acc,
            )

            hyp_f1_micro.append(f1_micro)
            hyp_f1_macro.append(f1_macro)
            hyp_f1_weight.append(f1_weight)
            hyp_acc.append(acc)
            hyp_precision.append(precesion)
            hyp_recall.append(recall)

        save_results_cc(
            output_path,
            params,
            hyp_f1_micro,
            hyp_f1_macro,
            hyp_f1_weight,
            hyp_acc,
            hyp_precision,
            hyp_recall,
        )


# for gaussian data and wordnet data
def run_train_UCI(params: dict, params_file):
    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(
        (params["output_path"] + "//" + params["method"])
    )
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logger(
        name=params["dataset_file"],
        format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s",
        reset=False,
        filepath=os.path.join(
            params["output_path"],
            (params["dataset_file"] + params["method"] + ".txt"),
        ),
    )

    archive_code(output_path, params_file)

    logger.info("%d GPUs available", torch.cuda.device_count())
    logger.info(
        "Using seed %d on class %d", params["seed"], params["class_label"]
    )

    for gassid in range(1, (params["data_num"] + 1)):
        params["class_label"] = gassid

        hyp_f1_micro = []
        hyp_f1_macro = []
        hyp_f1_weight = []
        hyp_acc = []
        hyp_recall = []
        hyp_precision = []

        for seedid in range(params["folds"]):
            params["seed"] = params["seed"] + 1
            train_x, train_y, test_x, test_y, val_x, val_y, space = (
                _build_datasets_UCI(params)
            )
            labels = np.unique(np.concatenate((train_y, test_y)))

            multi_class = True if len(labels) > 2 else False
            params["multi_class"] = multi_class

            logger.info(
                "%d train and %d test samples for fold %d",
                len(train_x),
                len(test_x),
                seedid,
            )
            if space == "euclidean":
                logger.info("Mapping to hyperbolic space")
                caredisk = PoincareBall(1)
                hyp_train_x = caredisk.expmap0(train_x, 1.0)
                hyp_test_x = caredisk.expmap0(test_x, 1.0)
                hyp_val_x = caredisk.expmap0(val_x, 1.0)
            else:
                hyp_train_x = train_x
                hyp_test_x = test_x
                hyp_val_x = val_x

            acc, f1_macro, f1_micro, f1_weight, precesion, recall = run_method(
                params,
                hyp_train_x,
                train_y,
                hyp_test_x,
                test_y,
                labels,
                train_x,
                test_x,
                val_x,
                hyp_val_x,
                val_y,
            )

            logger.info(
                "Hyperbolic tree f1 micro: %.4f, f1 macro: %.4f, f1_weight: %.4f. acc %.2f",
                f1_micro,
                f1_macro,
                f1_weight,
                acc,
            )

            hyp_f1_micro.append(f1_micro)
            hyp_f1_macro.append(f1_macro)
            hyp_f1_weight.append(f1_weight)
            hyp_acc.append(acc)
            hyp_precision.append(precesion)
            hyp_recall.append(recall)

        save_results_cc(
            output_path,
            params,
            hyp_f1_micro,
            hyp_f1_macro,
            hyp_f1_weight,
            hyp_acc,
            hyp_precision,
            hyp_recall,
        )


lossobject = {"poincare": customgobj, "original": logregobj}


def get_score(test_y, test_y_pred):
    f1_micro = f1_score(test_y, test_y_pred, average="micro")
    f1_macro = f1_score(test_y, test_y_pred, average="macro")
    f1_weight = f1_score(test_y, test_y_pred, average="weighted")
    acc = accuracy_score(test_y, test_y_pred)

    precesion = precision_score(
        test_y, test_y_pred, average="macro", zero_division=0
    )
    recall = recall_score(test_y, test_y_pred, average="macro", zero_division=0)
    return acc, f1_macro, f1_micro, f1_weight, precesion, recall


def getConfig(args=None):
    confini = {}
    confini["subsample"] = args["subsample"]
    confini["colsample_bylevel"] = args["colsample_bylevel"]
    confini["colsample_bynode"] = args["colsample_bynode"]
    confini["colsample_bytree"] = args["colsample_bytree"]
    confini["eta"] = args["eta"]
    confini["gamma"] = args["gamma"]
    return confini


class classifier_hyperxgb_EP:

    def __init__(self):
        print(f"create initial {0}")

    def selectOne(self, fitnessVal):
        max = sum([c for c in fitnessVal])
        pick = random.uniform(0, max)
        current = 0
        for id, chromosome in enumerate(fitnessVal):
            current += chromosome
            if current >= pick:
                return id  # we only have three options,


class classifier_hyperxgb:

    def init_(
        self,
        kclass=2,
        y_test=None,
        X_train=None,
        x_test=None,
        mode="original",
        seed=42,
    ):
        self.dtrain = X_train
        self.dtest = x_test
        self.y_test = y_test
        self.kClasses = kclass
        self.obj_native = lossobject[mode]
        self.seed = seed

    def train(
        self, config_dict1: Configuration, round=None
    ) -> tuple[float, Any, Any, Any]:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        self.kRounds = round  # -> num_boost_round
        custom_results = {}
        booster_custom = xgb.train(
            {
                "num_class": self.kClasses,
                "disable_default_eval_metric": True,
                "seed": self.seed,
                **config_dict1,
            },
            self.dtrain,
            num_boost_round=self.kRounds,
            obj=self.obj_native,
            custom_metric=accMetric,
            evals_result=custom_results,
            evals=[(self.dtest, "ttest")],
        )  # , early_stopping_rounds= 10

        # best_iteration = booster_custom.best_iteration
        predt_custom = booster_custom.predict(
            self.dtest
        )  # , iteration_range=(0, best_iteration)

        acc, f1_macro, f1_micro, f1_weight, precesion, recall = get_score(
            self.y_test, predt_custom
        )

        return acc, f1_macro, f1_micro, f1_weight, precesion, recall


class classifier_hyperxgb_EP_m(classifier_hyperxgb_EP):

    def init_pram(
        self,
        kclass=2,
        seed=42,
        train_x=None,
        train_y=None,
        test_x=None,
        test_y=None,
        eval_x=None,
        val_y=None,
    ):

        self.kClasses = kclass
        self.seed = seed
        self._ratio = 0.75
        self._downsize = 0.2
        self._upsize = 0.5
        self._steps = 5
        self._tryround = 5

        hy_train_x, hy_val_x, hy_train_y, hy_val_y = (
            train_x,
            eval_x,
            train_y,
            val_y,
        )
        self.dtrain = xgb.DMatrix(hy_train_x, hy_train_y)
        self.dtest = xgb.DMatrix(test_x, test_y)
        self.y_test = test_y
        self.dval = xgb.DMatrix(hy_val_x, hy_val_y)
        self.y_val = hy_val_y

    def train(
        self, config_dict1: Configuration, round=None
    ) -> tuple[float, Any, Any, Any, Any, Any]:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        self.kRounds = round  # -> num_boost_round
        if self._tryround > self.kRounds:
            self._tryround = self.kRounds
        FitnessSel = np.zeros([2])
        custom_results = {}

        booster_custom = xgb.train(
            {
                "num_class": self.kClasses,
                "disable_default_eval_metric": True,
                **config_dict1,
            },
            self.dtrain,
            num_boost_round=self._tryround,
            obj=logregobj,
            custom_metric=accMetric,
            evals_result=custom_results,
            evals=[(self.dval, "ttest")],
        )  # early_stopping_rounds= 10, here it is not working for that long

        FitnessSel[1] = custom_results.get("ttest").get("PyMError")[-1]
        poincare_results = {}
        booster_poincare = xgb.train(
            {
                "num_class": self.kClasses,
                "disable_default_eval_metric": True,
                **config_dict1,
            },
            self.dtrain,
            num_boost_round=self._tryround,
            obj=customgobj,
            custom_metric=accMetric,
            evals_result=poincare_results,
            evals=[(self.dval, "ttest")],
        )
        FitnessSel[0] = poincare_results.get("ttest").get("PyMError")[-1]

        modelobject = {"poincare": booster_poincare, "original": booster_custom}
        names = ["poincare", "original"]  # , 'hyper'

        bestsofar = np.min(FitnessSel)
        Fitness = np.argsort(FitnessSel)[::-1] + 1
        ids = np.argmax(Fitness)  # first choice is to use the best one
        booster_native = modelobject[names[ids]]
        obj_native = lossobject[names[ids]]

        ibins = np.arange(
            start=self._tryround, stop=self.kRounds, step=self._steps
        )
        early_stop = 0
        for idx in ibins:
            native_results = {}
            # Use the same objective function defined in XGBoost.
            booster_native = xgb.train(
                {
                    "num_class": self.kClasses,
                    "disable_default_eval_metric": True,
                    **config_dict1,
                },
                self.dtrain,
                num_boost_round=self._steps,
                obj=obj_native,
                custom_metric=accMetric,
                evals_result=native_results,
                evals=[(self.dval, "ttest")],
                xgb_model=booster_native,
            )
            # update the FitnessSel
            score = native_results.get("ttest").get("PyMError")[-1]
            if bestsofar <= score:
                Fitness[ids] = max(0, Fitness[ids] - self._downsize)
            else:
                bestsofar = score
                Fitness[ids] = Fitness[ids] + self._upsize
            ids = self.selectOne(
                Fitness
            )  # no need to select model again, only choose 'obj'
            obj_native = lossobject[names[ids]]

            # if score == native_results.get('ttest').get('PyMError')[0] and (early_stop == 0 or early_stop == 1):
            #     early_stop = early_stop + 1
            # else:  # continuous twice there is no improve on metric score, then we stop,
            #     early_stop = 0
            #
            # if early_stop == 2:
            #     break

        # it should have the last evaluation but we omit
        # itmax = self.kRounds - tryround

        predt_custom = booster_native.predict(self.dtest)

        acc, f1_macro, f1_micro, f1_weight, precesion, recall = get_score(
            self.y_test, predt_custom
        )

        return acc, f1_macro, f1_micro, f1_weight, precesion, recall


class classifier_hyperxgb_EP_v(classifier_hyperxgb_EP):

    def init_pram(
        self,
        kclass=2,
        seed=42,
        train_x=None,
        train_y=None,
        test_x=None,
        test_y=None,
        eval_x=None,
        val_y=None,
    ):
        self.kClasses = kclass
        self.seed = seed
        self._ratio = 0.85
        self._downsize = 0.2
        self._upsize = 0.3
        self._steps = 5

        # hy_train_x, hy_val_x, hy_train_y, hy_val_y = train_test_split(train_x, train_y, test_size=0.1,
        #                                                     random_state=42)
        hy_train_x, hy_val_x, hy_train_y, hy_val_y = (
            train_x,
            eval_x,
            train_y,
            val_y,
        )
        self.dtrain = xgb.DMatrix(hy_train_x, hy_train_y)
        self.dtest = xgb.DMatrix(test_x, test_y)
        self.y_test = test_y
        self.dval = xgb.DMatrix(hy_val_x, hy_val_y)
        self.y_val = hy_val_y

    def train(
        self, config_dict1: Configuration, round=None
    ) -> tuple[float, Any, Any, Any, Any, Any]:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""

        self._tryround = int(round / 2)
        FitnessSel = np.zeros([2])
        custom_results = {}

        booster_custom = xgb.train(
            {
                "num_class": self.kClasses,
                "disable_default_eval_metric": True,
                **config_dict1,
            },
            self.dtrain,
            num_boost_round=self._tryround,
            obj=logregobj,
            custom_metric=accMetric,
            evals_result=custom_results,
            evals=[(self.dval, "ttest")],
            early_stopping_rounds=10,
        )

        poincare_results = {}
        booster_poincare = xgb.train(
            {
                "num_class": self.kClasses,
                "disable_default_eval_metric": True,
                **config_dict1,
            },
            self.dtrain,
            num_boost_round=self._tryround,
            obj=customgobj,
            custom_metric=accMetric,
            evals_result=poincare_results,
            evals=[(self.dval, "ttest")],
            early_stopping_rounds=10,
        )

        predt_custom = predict(booster_custom, self.dval)
        predt_poin = predict(booster_poincare, self.dval)
        FitnessSel[1] = 1 - accuracy_score(self.y_val, predt_custom)
        FitnessSel[0] = 1 - accuracy_score(self.y_val, predt_poin)

        modelobject = {"poincare": booster_poincare, "original": booster_custom}
        names = ["poincare", "original"]  # , 'hyper'

        bestsofar = np.min(FitnessSel)
        Fitness = np.argsort(FitnessSel)[::-1] + 1
        ids = np.argmax(Fitness)  # first choice is to use the best one
        booster_native = modelobject[names[ids]]
        obj_native = lossobject[names[ids]]
        early_stop = 0
        ibins = np.arange(start=self._tryround, stop=round, step=self._steps)
        for idx, val in enumerate(ibins):
            native_results = {}
            # Use the same objective function defined in XGBoost.
            booster_native = xgb.train(
                {
                    "num_class": self.kClasses,
                    "disable_default_eval_metric": True,
                    **config_dict1,
                },
                self.dtrain,
                num_boost_round=self._steps,
                obj=obj_native,
                custom_metric=accMetric,
                evals_result=native_results,
                evals=[(self.dval, "ttest")],
                xgb_model=booster_native,
            )
            # update the FitnessSel
            predt_na = predict(booster_native, self.dval)
            native_v = 1 - accuracy_score(self.y_val, predt_na)

            if bestsofar <= native_v:
                Fitness[ids] = max(0, Fitness[ids] - self._downsize)
            else:
                bestsofar = native_v
                Fitness[ids] = Fitness[ids] + self._upsize
            ids = self.selectOne(
                Fitness
            )  # no need to select model again, only choose 'obj'
            obj_native = lossobject[names[ids]]

            # if native_results.get('ttest').get('PyMError')[0] == native_results.get('ttest').get('PyMError')[-1] and (early_stop == 0 or early_stop == 1):
            #     early_stop = early_stop + 1
            # else:  # continuous twice there is no improve on metric score, then we stop,
            #     early_stop = 0
            #
            # if early_stop == 2:
            #     break

        predt_custom = booster_native.predict(self.dtest)

        acc, f1_macro, f1_micro, f1_weight, precesion, recall = get_score(
            self.y_test, predt_custom
        )

        return acc, f1_macro, f1_micro, f1_weight, precesion, recall
