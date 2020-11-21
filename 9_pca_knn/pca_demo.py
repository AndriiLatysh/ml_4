import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
import sklearn.decomposition as sk_decomposition


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

demo_df = pd.read_csv("data/pca_demo_data.csv")

X = demo_df[["x", "y"]]
# plt.scatter(demo_df["x"], demo_df["y"])

standard_scaler = sk_preprocessing.StandardScaler()
X_standartized = standard_scaler.fit_transform(X)

ax1.scatter(X_standartized[:, 0], X_standartized[:, 1])

pca_transformer = sk_decomposition.PCA(n_components=2)
principal_components = pca_transformer.fit_transform(X_standartized)

ax2.scatter(principal_components[:, 0], principal_components[:, 1])

print(pca_transformer.components_)

plt.show()
