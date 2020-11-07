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

print("Tree")

diabetes_tree_model = sk_tree.DecisionTreeClassifier()
# diabetes_tree_model = sk_tree.DecisionTreeClassifier(max_depth=4)
diabetes_tree_model.fit(X_train, y_train)

tree_y_prediction = diabetes_tree_model.predict(X_test)

print("Accuracy: {}".format(sk_metrics.accuracy_score(y_test, tree_y_prediction)))

tree_confusion_matrix = sk_metrics.confusion_matrix(y_test, tree_y_prediction)
print(tree_confusion_matrix)

sk_tree.plot_tree(diabetes_tree_model, feature_names=column_names, class_names=["0", "1"], filled=True, rounded=True)

print()
# plt.show()

print("Forest")

diabetes_forest_model = sk_ensemble.RandomForestClassifier(n_jobs=-1)
diabetes_forest_model.fit(X_train, y_train)

forest_y_prediction = diabetes_forest_model.predict(X_test)

print("Accuracy: {}".format(sk_metrics.accuracy_score(y_test, forest_y_prediction)))

forest_confusion_matrix = sk_metrics.confusion_matrix(y_test, forest_y_prediction)
print(forest_confusion_matrix)
