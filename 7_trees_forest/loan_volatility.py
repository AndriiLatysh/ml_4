import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_tree


def convert_to_numeric_values(df):
    converted_df = df.copy()
    converted_df = converted_df.replace({"history": {"bad": 0, "fair": 1, "excellent": 2},
                                         "income": {"low": 0, "high": 1},
                                         "term": {3: 0, 10: 1},
                                         "risk": {"low": 0, "high": 1}})

    return converted_df


plt.figure(figsize=(12, 8))

loans_df = pd.read_csv("data/loans.csv")
numeric_loans_df = convert_to_numeric_values(loans_df)
print(numeric_loans_df)

feature_names = loans_df.columns.values[:-1]
X = numeric_loans_df[feature_names]
y = numeric_loans_df["risk"]

loan_decision_tree = sk_tree.DecisionTreeClassifier()
loan_decision_tree.fit(X, y)

sk_tree.plot_tree(loan_decision_tree, feature_names=feature_names, class_names=["low", "high"], filled=True, rounded=True)

plt.show()
