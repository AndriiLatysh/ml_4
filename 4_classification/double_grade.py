import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear_models
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection

qualifies_double_grade = pd.read_csv("data/double_grade.csv")

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade["qualifies"]

qualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 1]
unqualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 0]

plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="g")
plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="r")

plt.xlabel("Technical grade")
plt.ylabel("English grade")

number_of_folds = 4
cv_qualification_model = sk_linear_models.LogisticRegression()

cv_model_quality = sk_model_selection.cross_val_score(cv_qualification_model, X, y, cv=number_of_folds,
                                                      scoring="accuracy")
print(cv_model_quality)

cv_prediction = sk_model_selection.cross_val_predict(cv_qualification_model, X, y, cv=number_of_folds)
cv_confusion_matrix = sk_metrics.confusion_matrix(y, cv_prediction)
print(cv_confusion_matrix)

qualification_model = sk_linear_models.LogisticRegression()
qualification_model.fit(X, y)

modeled_qualification_probability = qualification_model.predict_proba(X)[:, 1]
qualifies_double_grade["modeled_probability"] = modeled_qualification_probability

pd.set_option("display.max_rows", None)
print(qualifies_double_grade.sort_values(by="modeled_probability"))

print(qualification_model.coef_)
print(qualification_model.intercept_)

plt.show()
