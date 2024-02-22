from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
import json


def make(X_train, y_train):

    def get_best_knn_params(X_train, y_train):
        parameters = {"kneighborsregressor__n_neighbors": [4, 7, 10],
                      "kneighborsregressor__weights": ["uniform", "distance"],
                      "kneighborsregressor__algorithm": ["auto", "ball_tree", "kd_tree", "brute"]}
        pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
        
        
        clf.fit(X_train, y_train)
        print("Got the best params", X_train.shape, y_train.shape)
        return clf.best_params_

    def get_best_tree_params(X_train, y_train):
        parameters = {"decisiontreeregressor__max_depth": [None, 5, 30, 50],
                      "decisiontreeregressor__min_samples_split": [5, 10, 20],
                      "decisiontreeregressor__min_samples_leaf": [4, 8, 16]}
        pipeline = make_pipeline(StandardScaler(), DecisionTreeRegressor())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)

        clf.fit(X_train, y_train)
        print("Got the best tree params", X_train.shape, y_train.shape)
        return clf.best_params_

    def get_best_forest_params(X_train, y_train):
        parameters = {"randomforestregressor__n_estimators": [100, 200, 400],
                      "randomforestregressor__max_depth": [None, 15, 20, 30],
                      "randomforestregressor__min_samples_split": [2, 10, 20],
                      "randomforestregressor__min_samples_leaf": [1, 4, 8]}
        pipeline = make_pipeline(StandardScaler(), RandomForestRegressor())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Got the best forest params", X_train.shape, y_train.shape)
        return clf.best_params_

    def get_best_svm_params(X_train, y_train):
        parameters = {"svr__kernel": ["poly", "rbf", "sigmoid"],
                      "svr__degree": [4, 5, 6],
                      "svr__C": [10, 100, 1000]}
        pipeline = make_pipeline(StandardScaler(), SVR())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Got the best svm params", X_train.shape, y_train.shape)
        return clf.best_params_

    def get_best_lin_reg_params(X_train, y_train):
        parameters = {"linearregression__fit_intercept": [True, False]}
        pipeline = make_pipeline(StandardScaler(), LinearRegression())
        clf = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Got the best lin reg params", X_train.shape, y_train.shape)
        return clf.best_params_

    knn_params = get_best_knn_params(X_train, y_train)
    tree_params = get_best_tree_params(X_train, y_train)
    forest_params = get_best_forest_params(X_train, y_train)
    svm_params = get_best_svm_params(X_train, y_train)
    lin_reg_params = get_best_lin_reg_params(X_train, y_train)

    full_params = {"knn": knn_params, "tree": tree_params, "forest": forest_params, "svm": svm_params, "lin_reg": lin_reg_params}

    with open("best_params.json", "w") as file:
        json.dump(full_params, file)