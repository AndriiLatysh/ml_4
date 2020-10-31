import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as sk_svm
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import double_grade_svm_utility as svm_utility

qualifies_double_grade = pd.read_csv("data/double_grade_reevaluated.csv")

svm_utility.plot_values(qualifies_double_grade)

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade["qualifies"]

# svm_soft_non_linear_classifier = sk_svm.SVC(kernel="rbf")
# svm_soft_non_linear_classifier.fit(X, y)
# svm_utility.plot_model(svm_soft_non_linear_classifier)

parameter_grid = {"kernel": ["rbf"],
                  "C": [10 ** p for p in range(-2, 6)],
                  "gamma": [10 ** p for p in range(-6, 2)]}

grid_search = sk_model_selection.GridSearchCV(sk_svm.SVC(),
                                              param_grid=parameter_grid,
                                              cv=4)
grid_search.fit(X, y)

print(grid_search.best_params_)

modeled_qualification = grid_search.predict(X)
confusion_matrix = sk_metrics.confusion_matrix(y, modeled_qualification)

print(confusion_matrix)

svm_utility.plot_model(grid_search.best_estimator_)

plt.show()
