import pandas as pd
import numpy as np
import sklearn.model_selection as sklearn_model_selection
import sklearn.linear_model as sklearn_linear
import sklearn.feature_extraction.text as sklearn_text
import sklearn.metrics as sklearn_metrics


imdb_reviews = pd.read_csv("data/imdb_dataset_prepared.csv")

X = imdb_reviews["review"]
y = imdb_reviews["sentiment"]

print("Vectorisation starting.")
tfidf = sklearn_text.TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
X = tfidf.fit_transform(X)
print("Vectorisation finished.")

# print(X)
# print(X.shape)

X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)

print("Logistic Regression training starting.")
logistic_model = sklearn_linear.LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
print("Logistic Regression training finished.")

y_predicted = logistic_model.predict(X_test)

print("Accuracy: {:.2f} %".format(sklearn_metrics.accuracy_score(y_test, y_predicted) * 100))
print(sklearn_metrics.confusion_matrix(y_test, y_predicted))

top_phrase_count = 20
tfidf_feature_names = tfidf.get_feature_names()

print("Negative:")
top_negative_phrases_indexes = np.argsort(logistic_model.coef_[0])[:top_phrase_count]
top_negative_phrases = [tfidf_feature_names[z] for z in top_negative_phrases_indexes]
print(top_negative_phrases)

print("Positive:")
top_positive_phrases_indexes = np.argsort(logistic_model.coef_[0])[-top_phrase_count:]
top_positive_phrases = [tfidf_feature_names[z] for z in top_positive_phrases_indexes]
print(top_positive_phrases)
