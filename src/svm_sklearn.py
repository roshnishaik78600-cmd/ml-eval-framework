
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load data
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn", "customerID"])

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    ), make_column_selector(dtype_include="number")),

    ("cat", make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    ), make_column_selector(dtype_exclude="number"))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Linear SVM
model = make_pipeline(
    preprocessor,
    LinearSVC(C=1.0, max_iter=5000, random_state=42)
)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
scores = model.decision_function(X_test)

print("Linear SVM Results")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, scores), 4))
print(classification_report(
    y_test, pred, target_names=["No Churn", "Churn"]
))
