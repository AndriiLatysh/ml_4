import pandas as pd
import numpy as np
import sklearn.linear_model as sk_linear_models
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection


def model_to_string(model, labels, precision=4):
    model_str = "{} = ".format(labels[-1])
    for z in range(len(labels) - 1):
        model_str += "{} * {} + ".format(round(model.coef_.flatten()[z], precision), labels[z])
    model_str += "{}".format(round(model.intercept_[0], precision))
    return model_str


def train_linear_model(X, y):
    linear_regression = sk_linear_models.LinearRegression()
    linear_regression.fit(X, y)
    return linear_regression


def get_MSE(model, X, y_true):
    y_predicted = model.predict(X)
    MSE = sk_metrics.mean_squared_error(y_true, y_predicted)
    return MSE


advertising_data = pd.read_csv("data/advertising.csv", index_col=0)
print(advertising_data)

ad_data = advertising_data[["TV", "radio", "newspaper"]]
sales_data = advertising_data[["sales"]]

linear_regression = sk_linear_models.LinearRegression()
lasso_regression = sk_linear_models.Lasso()
ridge_regression = sk_linear_models.Ridge()

linear_regression.fit(ad_data, sales_data)
lasso_regression.fit(ad_data, sales_data)
ridge_regression.fit(ad_data, sales_data)

labels = advertising_data.columns.values

print("Linear regression.")
print(model_to_string(linear_regression, labels))
print()

print("Ridge regression (L2).")
print(model_to_string(ridge_regression, labels))
print()

print("Lasso regression (L1).")
print(model_to_string(lasso_regression, labels))
print()

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(ad_data, sales_data, shuffle=True)

linear_regression = train_linear_model(X_train, y_train)
print(model_to_string(linear_regression, labels))
print("Train MSE = {}".format(get_MSE(linear_regression, X_train, y_train)))
print("Test MSE = {}".format(get_MSE(linear_regression, X_test, y_test)))
print()

for z in range(len(labels) - 1):
    feature_name = labels[z]
    print("{} removed:".format(feature_name))

    X_train_2_features = X_train.drop(feature_name, axis=1)
    X_test_2_features = X_test.drop(feature_name, axis=1)
    labels_2_features = np.delete(labels, z)

    model_2_features = train_linear_model(X_train_2_features, y_train)
    print(model_to_string(model_2_features, labels_2_features))
    print("Test MSE = {}".format(get_MSE(model_2_features, X_test_2_features, y_test)))
    print()
