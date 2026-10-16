
import pandas as pd
import optuna

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"].map({"Yes": 1, "No": 0})

def objective(trial):
    C = trial.suggest_float("C", 0.01, 10)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=1000)
    )

    cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=42
    )

    score = cross_val_score(
        model, X, y, cv=cv, scoring="roc_auc"
    ).mean()

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best ROC-AUC:", round(study.best_value, 4))
print("Best C:", study.best_params["C"])
