import pandas as pd
from sklearn.model_selection import train_test_split
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

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print(model.coef_)
    print(model.intercept_)
    print(model.predict_proba(X_test[:5]))
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.2%}")

def des_tree_model(data):
    data = pd.DataFrame(data)

    labels = [0] * 500 + [1] * 500

    data["cheating"] = labels

    X = data.drop(columns="cheating")
    y = data["cheating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print(model.feature_importances_)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.2%}")

def rand_forest_model(data):
    data = pd.DataFrame(data)

    labels = [0] * 500 + [1] * 500

    data["cheating"] = labels

    X = data.drop(columns="cheating")
    y = data["cheating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print(model.feature_importances_)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.2%}")