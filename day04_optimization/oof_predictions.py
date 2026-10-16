
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"].map({"Yes": 1, "No": 0})

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = cross_val_predict(
    model, X, y, cv=cv, method="predict_proba"
)[:, 1]

pd.DataFrame({
    "actual": y,
    "oof_probability": oof
}).to_csv("day04_optimization/oof_predictions.csv", index=False)

print("OOF predictions created:", len(oof))
print("Saved to day04_optimization/oof_predictions.csv")
