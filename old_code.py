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