
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn", "customerID"])

# Add one engineered feature
X["ServiceCount"] = (
    df[["PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies"]] == "Yes"
).sum(axis=1)

# Same train/test split for both experiments
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def train_model(features, name):
    num_cols = features.select_dtypes(include="number").columns
    cat_cols = features.select_dtypes(exclude="number").columns

    prep = ColumnTransformer([
        ("num", make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler()
        ), num_cols),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        ), cat_cols)
    ])

    model = make_pipeline(prep, LogisticRegression(max_iter=1000))
    model.fit(features.loc[X_train.index], y_train)

    pred = model.predict(features.loc[X_test.index])
    prob = model.predict_proba(features.loc[X_test.index])[:, 1]

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

# Compare with and without the new feature
train_model(X.drop(columns=["ServiceCount"]), "Original Features")
train_model(X, "With ServiceCount")
