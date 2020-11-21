import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as sk_neighbours
import sklearn.model_selection as sk_model_selection


def plot_model(model, qualifies_double_grade_df):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    max_grade = 101
    prediction_points = []

    for english_grade in range(max_grade):
        for technical_grade in range(max_grade):
            prediction_points.append([technical_grade, english_grade])

    probability_levels = model.predict_proba(prediction_points)[:, 1]
    probability_matrix = probability_levels.reshape(max_grade, max_grade)

    plt.contourf(probability_matrix, cmap="rainbow")  # cmap="RdYlBu"/"binary"

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualified_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualified_double_grade_df[["technical_grade", "english_grade"]]
y = qualified_double_grade_df["qualifies"]

for k in range(1, 10, 2):
    print(f"{k} neighbours:")

    double_grade_knn_model = sk_neighbours.KNeighborsClassifier(n_neighbors=k)
    cv_double_grade_model_quality = sk_model_selection.cross_val_score(double_grade_knn_model, X, y, cv=4,
                                                                       scoring="accuracy")
    print("Accuracy: {}".format(np.mean(cv_double_grade_model_quality)))

double_grade_knn_model = sk_neighbours.KNeighborsClassifier(n_neighbors=3)
double_grade_knn_model.fit(X, y)

plot_model(double_grade_knn_model, qualified_double_grade_df)

plt.show()
