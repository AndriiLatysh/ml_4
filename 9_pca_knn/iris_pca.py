import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
import sklearn.decomposition as sk_decomposition
import sklearn.model_selection as sk_model_selection
import sklearn.linear_model as sk_linear

iris_df = pd.read_csv("data/iris.csv").sample(frac=1)
columns_names = iris_df.columns.tolist()

X = iris_df[columns_names[:-1]]
y = iris_df[columns_names[-1]]

standard_scaler = sk_preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)

laber_encoder = sk_preprocessing.LabelEncoder()
y = laber_encoder.fit_transform(y)

cv_iris_log_model = sk_linear.LogisticRegression()
cv_iris_log_model_quality = sk_model_selection.cross_val_score(cv_iris_log_model, X, y, cv=4, scoring="accuracy")

print("Model quality:")
print(np.mean(cv_iris_log_model_quality))

pca_2_components = sk_decomposition.PCA(n_components=2)
principal_components = pca_2_components.fit_transform(X)

# plt.scatter(x=principal_components[:, 0], y=principal_components[:, 1], c=y, cmap="prism")

cv_iris_2_log_model = sk_linear.LogisticRegression()
cv_iris_2_log_model_quality = sk_model_selection.cross_val_score(cv_iris_2_log_model, principal_components, y, cv=4,
                                                                 scoring="accuracy")

print("2 pc model quality:")
print(np.mean(cv_iris_2_log_model_quality))

pca_all_components = sk_decomposition.PCA()
pca_all_components.fit(X)

print(pca_all_components.explained_variance_)
print("Explained variance ratio:")
print(pca_all_components.explained_variance_ratio_)

components = list(range(1, pca_all_components.n_components_ + 1))
plt.plot(components, np.cumsum(pca_all_components.explained_variance_ratio_), marker="o")
plt.xlabel("Number of components")
plt.ylabel("Explained variance")
plt.ylim(0, 1.1)

plt.show()
