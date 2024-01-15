# Imports
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# Stylometry
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler

# Imports confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def split_data():
    """ Read and split the raw data to X_train, X_test, y_train, y_test. Only keeping posts of nationalities
     with more than 2000 posts."""

    data = pd.read_csv('data/preprocessed_data.csv').drop(columns=['Unnamed: 0'])
    #data = data[:2000]# Smaller data for testing with faster running time
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

def naivebayes(X_train, X_test, y_train, y_test):
    """ Train and predict the Naive Bayes model with TF-IDF features"""
    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the classifier Naive Bayes using the TF-IDF features
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test_tfidf)
    return y_test, y_pred

def svm_model1(X_train, X_test, y_train, y_test):
    """ Train and predict the SVM model using TF-IDF features"""
    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the SVM model using the TF-IDF features
    classifier.fit(X_train_tfidf, y_train)

    # Make predictions on the test se
    y_pred = classifier.predict(X_test_tfidf)
    return y_test, y_pred

def extract_stylometric_features(post):
    """ Extract the stylometriuc features of the reddit posts given as input, extracts 9 stylometric features"""

    # Tokenize post
    words = word_tokenize(post)
    sentences = sent_tokenize(post)

    # Feature 1: Average word length
    average_word_length = np.mean([len(word) for word in words])

    # Feature 2: Average sentence length
    average_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])

    # Feature 3: Percentage of stopwords
    stop_words = set(stopwords.words('english'))
    stopwords_percentage = len([word for word in words if word.lower() in stop_words]) / len(words) * 100

    # Feature 4: Number of nouns
    pos_tags = pos_tag(words)
    num_nouns = len([word for word, pos in pos_tags if pos.startswith('N')])

    # Feature 5: Number of verbs
    num_verbs = len([word for word, pos in pos_tags if pos.startswith('V')])

    # Feature 6: Number of adjectives
    num_adjectives = len([word for word, pos in pos_tags if pos.startswith('J')])

    # Feature 7: Number of adverbs
    num_adverbs = len([word for word, pos in pos_tags if pos.startswith('R')])

    # Feature 8: Percentage of uppercase words
    uppercase_words_percentage = len([word for word in words if word.isupper()]) / len(words) * 100

    # Feature 9: Average syllables per word
    syllables_per_word = np.mean([sum(char.isdigit() for char in word) for word in words])

    # Feature 10: Total number of words
    num_words = len(words)

    # Feature 11: Vocabulary richness
    vocab = list(set(words))
    vocab_richness = len(vocab)/num_words

    # Feature 12: Vocabulary size
    vocab_size = len(vocab)

    # Feature 13: Frequency of "."
    freq_dot = len([word for word in words if word == "."])

    # Feature 14: Frequency of ","
    freq_comma = len([word for word in words if word == ","])

    # Feature 15: Frequency of ":"
    freq_colon = len([word for word in words if word == ":"])

    # Feature 16: Frequency of "?"
    freq_question = len([word for word in words if word == "?"])

    # Feature 17: Frequency of "?"
    freq_exclamation = len([word for word in words if word == "!"])

    # Feature 18: Standard deviation sentence length
    std_dev_sentence_length = np.std([len(sentence.split()) for sentence in sentences])

    # Feature 19: Standard deviation sentence length
    std_dev_word_length = np.std([len(word) for word in words])

    return [average_word_length, average_sentence_length, stopwords_percentage,
            num_nouns, num_verbs, num_adjectives, num_adverbs,
            uppercase_words_percentage, syllables_per_word, num_words, vocab_richness, vocab_size, freq_dot,
            freq_comma, freq_colon, freq_question, freq_exclamation, std_dev_sentence_length, std_dev_word_length]

def svm_model2(X_train, X_test, y_train, y_test):
    """ Train and predict the SVM model using stylometric features"""
    # Extract features and labels
    X_train_stylometric = [extract_stylometric_features(post) for post in X_train]
    X_test_stylometric = [extract_stylometric_features(post) for post in X_test]

    # Standardize features
    scaler = StandardScaler()
    X_train_stylometric = scaler.fit_transform(X_train_stylometric)
    X_test_stylometric = scaler.transform(X_test_stylometric)

    classifier = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the SVM model using the TF-IDF features
    classifier.fit(X_train_stylometric, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test_stylometric)
    return y_test, y_pred

def combination_model(X_train, X_test, y_train, y_test):

    # Extract features and labels
    y_test_tfidf, y_pred_tfidf = svm_model1(X_train, X_test, y_train, y_test)

    # Model 2: Stylometry model
    y_test_stylometric, y_pred_stylometric = svm_model2(X_train, X_test, y_train, y_test)

    # Combine predictions
    X_combined = np.column_stack((y_pred_tfidf, y_pred_stylometric))

    # Final Classifier
    final_classifier = LogisticRegression()
    final_classifier.fit(X_combined, y_train)

    # Make predictions on the test set
    X_test_combined = np.column_stack((y_test_tfidf, y_test_stylometric))
    y_pred_combined = final_classifier.predict(X_test_combined)

    return y_test, y_pred_combined

def metrics(y_test, y_pred):
    """ Compute the requiured metrics to evaluate the model's performance
    - Accuracy
    - Macro averaged precision
    - Macro averaged recall
    - Macro averaged f-score
    """

    accuracy = accuracy_score(y_test, y_pred)
    macro_precision, macro_recall, macro_f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=1)

    return accuracy, macro_precision, macro_recall, macro_f

def multiclass_confusion_matrix(y_test, y_pred, model):
    labels = list(set(y_test))
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(len(labels), len(labels)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Multiclass Confusion Matrix of {model}')
    plt.savefig(f"cm_{model}.png")


# Initiate results table
results = pd.DataFrame(columns=['Model','Accuracy','Precision','Recall','F-score','Test-size'])
def test_model(results, X_train, X_test, y_train, y_test, model):
    """ Test the model and add statistics to the results table."""
    if model == "Majority-baseline":
        y_test, y_pred = majority_baseline(X_train, X_test, y_train, y_test)
    elif model == "Naive Bayes":
        y_test, y_pred = naivebayes(X_train, X_test, y_train, y_test)
    elif model == "SVM_tf_idf":
        y_test, y_pred = svm_model1(X_train, X_test, y_train, y_test)
    elif model == "SVM_stylometry":
        y_test, y_pred = svm_model2(X_train, X_test, y_train, y_test)
    elif model == 'Combined_model':
        y_test, y_pred = combination_model(X_train, X_test, y_train, y_test)

    # Get the performance metrics and add to the resuts table
    accuracy, macro_precision, macro_recall, macro_f = metrics(y_test, y_pred)
    majority_resuts = {'Model': model, 'Accuracy': accuracy,
                       'Precision': macro_precision, 'Recall': macro_recall,
                       'F-score': macro_f, 'Test-size': len(y_test)}
    results = results.append(majority_resuts, ignore_index=True)

    # Creating a confusion matrix of the prediction
    multiclass_confusion_matrix(y_test, y_pred, model)

    # Creatinng and saving a DataFrame prediction values for later use
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    df.to_csv(f'predictions_{model}.csv', index=False)
    return results



# # Get data
# X_train, X_test, y_train, y_test = split_data()
#
# results = test_model(results, X_train, X_test, y_train, y_test, "Majority-baseline")
# #results = test_model(results, X_train, X_test, y_train, y_test, "Naive Bayes")
# results = test_model(results, X_train, X_test, y_train, y_test, "SVM_tf_idf")
# results = test_model(results, X_train, X_test, y_train, y_test, "SVM_stylometry")
# results = test_model(results, X_train, X_test, y_train, y_test, "Combined_model")
# print(results)
style = pd.read_csv('predictions_SVM_stylometry.csv')
print('Stylometry:', metrics(style['y_test'], style['y_pred']))
major = pd.read_csv('predictions_Majority-baseline.csv')
print('Majority baseline:', metrics(major['y_test'], major['y_pred']))
tfidf = pd.read_csv('predictions_SVM_tf_idf.csv')
print('TF-IDF:', metrics(tfidf['y_test'], tfidf['y_pred']))
# print(X_train[0])
# print(extract_stylometric_features(X_train[0]))
# print(extract_stylometric_features(X_train[1]))
