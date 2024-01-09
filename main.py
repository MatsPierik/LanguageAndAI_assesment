# Imports
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB




def split_data():
    """ Read and split the raw data to X_train, X_test, y_train, y_test. Only keeping posts of nationalities
     with more than 2000 posts."""

    data = pd.read_csv("data/nationality.csv")

    # Get only the posts of a nationality with more than 2000 posts
    data = data.groupby('nationality').filter(lambda x: len(x) > 2000)

    X = data["post"]
    y = data['nationality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test



def majority_baseline(X_train, X_test, y_train, y_test):
    """ Get the majority baseline predicted values"""
    y_pred = y_test.copy()
    y_pred[:] = mode(y_train)
    return y_test, y_pred

def tf_idf(X_train, X_test, y_train, y_test):
    # Create the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test data
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train a classifier (e.g., Naive Bayes) using the TF-IDF features
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test_tfidf)
    return y_test, y_pred

def metrics(y_test, y_pred):
    """ Compute the requiured metrics to evaltuate the model's performance
    - Accuracy
    - Macro averaged precision
    - Macro averaged recall
    - Macro averaged f-score
    """

    accuracy = accuracy_score(y_test, y_pred)
    macro_precision, macro_recall, macro_f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=1)

    return accuracy, macro_precision, macro_recall, macro_f


# Initiate results table
results = pd.DataFrame(columns=['Model','Accuracy','Precision','Recall','F-score','Test-size'])
def test_model(results, X_train, X_test, y_train, y_test, model):
    """ Test the model and add statistics to the results table"""
    if model == "Majority-baseline":
        y_test, y_pred = majority_baseline(X_train, X_test, y_train, y_test)
    elif model == "tf*idf":
        y_test, y_pred = tf_idf(X_train, X_test, y_train, y_test)

    accuracy, macro_precision, macro_recall, macro_f = metrics(y_test, y_pred)
    majority_resuts = {'Model': model, 'Accuracy': accuracy,
                       'Precision': macro_precision, 'Recall': macro_recall,
                       'F-score': macro_f, 'Test-size': len(y_test)}
    results = results.append(majority_resuts, ignore_index=True)
    return results



# Get data
X_train, X_test, y_train, y_test = split_data()

results = test_model(results, X_train, X_test, y_train, y_test, "Majority-baseline")
results = test_model(results, X_train, X_test, y_train, y_test, "tf*idf")
print(results)