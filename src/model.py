import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest

def log_reg_model(data):
    data = pd.DataFrame(data)

    labels = [0] * 500 + [1] * 500

    data["cheating"] = labels

    X = data.drop(columns="cheating")
    y = data["cheating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Found that C=0.01 gives on average best accuracy
    model = LogisticRegression(C=0.01, max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Logistic Regression Accuracy = {accuracy_score(y_test, predictions):.2%}")

def des_tree_model(data):   
    data = pd.DataFrame(data)

    labels = [0] * 500 + [1] * 500

    data["cheating"] = labels

    X = data.drop(columns="cheating")
    y = data["cheating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Found that max_depth=5 gives on average best accuracy
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Decision Tree Accuracy = {accuracy_score(y_test, predictions):.2%}")

def rand_forest_model(data):
    data = pd.DataFrame(data)

    labels = [0] * 500 + [1] * 500

    data["cheating"] = labels

    X = data.drop(columns="cheating")
    y = data["cheating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Found that 50 estimators give on average best accuracy 
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Random Forest Accuracy = {accuracy_score(y_test, predictions):.2%}")

def iso_forest_model(fair_data, cheating_data):
    # Unsupervised learning model to detect cheating sessions
    model = IsolationForest(contamination=0.2, random_state=42)

    # Split the fair data into training and testing sets 
    fair_train, fair_test = train_test_split(fair_data, test_size=0.2, random_state=42)
    
    # Fit the model on fair data training set
    model.fit(fair_train)

    # Predict on fair testing set and cheating data
    fair_predictions = model.predict(fair_test)
    cheating_predictions = model.predict(cheating_data)

    # Calculate the number of anomalies detected in fair and cheating sessions
    fair_anomalies = sum(pred == -1 for pred in fair_predictions)
    cheating_anomalies = sum(pred == -1 for pred in cheating_predictions)

    # Print fair false positive rate out of the testing set
    print(f"False positive fair sessions: {fair_anomalies} out of {len(fair_test)}")
    print(f"False positive rate: {fair_anomalies / len(fair_test):.2%}")
    # Print detection rate out of the cheating data
    print(f"Cheating sessions detected: {cheating_anomalies} out of {len(cheating_data)}")
    print(f"Detection rate: {cheating_anomalies / len(cheating_data):.2%}")

    # Calculate total accuracy of the model
    total_correct = len(fair_test) - fair_anomalies
    cheating_correct = cheating_anomalies

    total_accuracy = (total_correct + cheating_correct) / (len(fair_test) + len(cheating_data))
    print(f"Total isolation forest accuracy: {total_accuracy:.2%}")
