from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

def train_models(X_train, X_test, y_train, y_test):
    results = {}

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    results['Logistic Regression'] = {
        "model": lr,
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr),
        "report": classification_report(y_test, y_pred_lr)
    }

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    results['Decision Tree'] = {
        "model": dt,
        "accuracy": accuracy_score(y_test, y_pred_dt),
        "confusion_matrix": confusion_matrix(y_test, y_pred_dt),
        "report": classification_report(y_test, y_pred_dt)
    }

    return results

def save_model(model, filename="saved_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)