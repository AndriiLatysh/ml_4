import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_tree
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.ensemble as sk_ensemble


plt.figure(figsize=(18, 10))

diabetes_df = pd.read_csv("data/pima-indians-diabetes.csv")
column_names = diabetes_df.columns.values

X = diabetes_df[column_names[:-1]]
y = diabetes_df[column_names[-1]]

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y)

adaboost_classifier = sk_ensemble.AdaBoostClassifier()
adaboost_classifier.fit(X_train, y_train)

y_predicted = adaboost_classifier.predict(X_test)

print("Accuracy: {}".format(sk_metrics.accuracy_score(y_test, y_predicted)))

confusion_matrix = sk_metrics.confusion_matrix(y_test, y_predicted)
print(confusion_matrix)
