import pandas as pd
import re
import string
import nltk


imdb_reviews = pd.read_csv("data/IMDB Dataset.csv")

X = imdb_reviews["review"].iloc[3]
print(X, "\n")

X = re.sub("<.*?>", " ", X)
X = X.lower()
X = X.translate(str.maketrans("", "", string.punctuation))

# stemmer = nltk.PorterStemmer()
# stemmer = nltk.LancasterStemmer()
lemmatizer = nltk.WordNetLemmatizer()

X = nltk.word_tokenize(X)

# X = [stemmer.stem(word) for word in X]
X = [lemmatizer.lemmatize(word) for word in X]

X = " ".join(X)

print(X)
