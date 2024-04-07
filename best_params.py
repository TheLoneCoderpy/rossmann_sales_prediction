from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import json


def make(X_train, y_train):
    def get_best_knn_params(X_train, y_train):
        parameters = {"kneighborsregressor__n_neighbors": [4, 6],
                      "kneighborsregressor__weights": ["uniform", "distance"],
                      "kneighborsregressor__algorithm": ["auto", "ball_tree"]}
        pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
        
        clf.fit(X_train, y_train)
        print("Got the best params", X_train.shape, y_train.shape)
        return clf.best_params_

    def get_best_tree_params(X_train, y_train):
        parameters = {"decisiontreeregressor__max_depth": [30, 50],
                      "decisiontreeregressor__min_samples_split": [5, 10],
                      "decisiontreeregressor__min_samples_leaf": [4, 8]}
        pipeline = make_pipeline(StandardScaler(), DecisionTreeRegressor())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)

        clf.fit(X_train, y_train)
        print("Got the best tree params", X_train.shape, y_train.shape)
        return clf.best_params_

    def get_best_forest_params(X_train, y_train):
        parameters = {"randomforestregressor__n_estimators": [200, 400],
                      "randomforestregressor__max_depth": [15, 25],
                      "randomforestregressor__min_samples_split": [2, 10],
                      "randomforestregressor__min_samples_leaf": [1, 4]}
        pipeline = make_pipeline(StandardScaler(), RandomForestRegressor())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Got the best forest params", X_train.shape, y_train.shape)
        return clf.best_params_

    #def get_best_svm_params(X_train, y_train):
    #     parameters = {"svr__kernel": ["rbf", "sigmoid"],
    #                   "svr__degree": [4, 5],
    #                   "svr__C": [1000, 1200]}
    #     pipeline = make_pipeline(StandardScaler(), SVR())
    #     clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
    #     clf.fit(X_train, y_train)
    #     print("Got the best svm params", X_train.shape, y_train.shape)
    #     return clf.best_params_

    #def get_best_lin_reg_params(X_train, y_train):
    #    parameters = {"linearregression__fit_intercept": [True, False]}
    #    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    #    clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
    #    clf.fit(X_train, y_train)
    #    print("Got the best lin reg params", X_train.shape, y_train.shape)
    #    return clf.best_params_

    def get_best_xg_params(X_train, y_train):
        parameters = {"xgbregressor__n_estimators": [100, 200, 400],
                      "xgbregressor__max_depth": [3, 5, 7],
                      "xgbregressor__learning_rate": [0.05, 0.1, 0.2]}
        pipeline = make_pipeline(StandardScaler(), XGBRegressor())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Got the best xg params", X_train.shape, y_train.shape)
        return clf.best_params_
    
    def get_best_adaboost_params(X_train, y_train):
        parameters = {"adaboostregressor__n_estimators": [50, 100, 200],
                    "adaboostregressor__learning_rate": [0.01, 0.1, 1]}
        pipeline = make_pipeline(StandardScaler(), AdaBoostRegressor())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Got the best adaboost params", X_train.shape, y_train.shape)
        return clf.best_params_

    def get_best_gradientboost_params(X_train, y_train):
        parameters = {"gradientboostingregressor__n_estimators": [100, 200, 400],
                    "gradientboostingregressor__learning_rate": [0.01, 0.1, 1],
                    "gradientboostingregressor__max_depth": [3, 5, 7]}
        pipeline = make_pipeline(StandardScaler(), GradientBoostingRegressor())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Got the best gradientboost params", X_train.shape, y_train.shape)
        return clf.best_params_

    knn_params = get_best_knn_params(X_train, y_train)
    tree_params = get_best_tree_params(X_train, y_train)
    forest_params = get_best_forest_params(X_train, y_train)
    svm_params = get_best_svm_params(X_train, y_train)
    #lin_reg_params = get_best_lin_reg_params(X_train, y_train)
    xg_params = get_best_xg_params(X_train, y_train)
    adaboost_params = get_best_adaboost_params(X_train, y_train)
    gradientboost_params = get_best_gradientboost_params(X_train, y_train)


    full_params = {"knn": knn_params, 
                   "tree": tree_params, 
                   "forest": forest_params, 
                   "svm": svm_params, 
                   "lin_reg": lin_reg_params, 
                   "xg": xg_params,
                   "adaboost": adaboost_params,
                   "gradientboost": gradientboost_params}

    with open("best_params.json", "w") as file:
        json.dump(full_params, file)