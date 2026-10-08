import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, isolation_forest

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
    
    scores = cross_val_score(model, X, y, cv=5)

    print(scores)
    print(f"Mean Accuracy: {scores.mean():.2%}")   

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

    scores = cross_val_score(model, X, y, cv=5)

    print(scores)
    print(f"Mean Accuracy: {scores.mean():.2%}")

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

    scores = cross_val_score(model, X, y, cv=5)

    print(scores)
    print(f"Mean Accuracy: {scores.mean():.2%}")

def isolation_forest_model(data):
    # Unsupervised learning model to detect cheating sessions
    data = pd.DataFrame(data)

