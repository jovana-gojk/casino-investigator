import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def log_reg_model(data):
    data = pd.DataFrame(data)

    labels = [0] * 500 + [1] * 500

    data["cheating"] = labels

    X = data.drop(columns="cheating")
    y = data["cheating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"Logistic Regression with C={C}: Accuracy = {accuracy_score(y_test, predictions):.2%}")

def des_tree_model(data):
    data = pd.DataFrame(data)

    labels = [0] * 500 + [1] * 500

    data["cheating"] = labels

    X = data.drop(columns="cheating")
    y = data["cheating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for max_depth in [1, 2, 3, 5, 10, 15, None]:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"Decision Tree with max_depth={max_depth}: Accuracy = {accuracy_score(y_test, predictions):.2%}")

def rand_forest_model(data):
    data = pd.DataFrame(data)

    labels = [0] * 500 + [1] * 500

    data["cheating"] = labels

    X = data.drop(columns="cheating")
    y = data["cheating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for n_estimators in [10, 20, 50, 100]:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"Random Forest with {n_estimators} estimators: Accuracy = {accuracy_score(y_test, predictions):.2%}")