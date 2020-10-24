import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as sk_svm
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import double_grade_svm_utility as svm_utility

qualifies_double_grade = pd.read_csv("data/double_grade.csv")

svm_utility.plot_values(qualifies_double_grade)

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade["qualifies"]

number_of_folds = 4
cv_qualification_model = sk_svm.SVC(kernel="linear")

cv_model_quality = sk_model_selection.cross_val_score(cv_qualification_model, X, y, cv=number_of_folds,
                                                      scoring="accuracy")
print(cv_model_quality)

cv_prediction = sk_model_selection.cross_val_predict(cv_qualification_model, X, y, cv=number_of_folds)
cv_confusion_matrix = sk_metrics.confusion_matrix(y, cv_prediction)
print(cv_confusion_matrix)

qualification_model = sk_svm.SVC(kernel="linear")
qualification_model.fit(X, y)

print(qualification_model.coef_)
print(qualification_model.intercept_)

svm_utility.plot_model(qualification_model)

plt.show()
