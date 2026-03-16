import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ======================================
# SPLIT TRAIN TEST (tu lógica ANI)
# ======================================

def crear_train_test(df):

    PHAGE_COL = "PHAGE"
    GROUP_COL = "FAGOS_ANI"

    df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()
    df[PHAGE_COL] = df[PHAGE_COL].astype(str).str.strip()

    unique_phages = df[[PHAGE_COL, GROUP_COL]].drop_duplicates()

    test_phages = (
        unique_phages
        .groupby(GROUP_COL)
        .filter(lambda x: x[PHAGE_COL].nunique() > 3)
        .groupby(GROUP_COL)
        .sample(1, random_state=42)
    )

    test_set = df[df[PHAGE_COL].isin(test_phages[PHAGE_COL])].copy()
    train_set = df[~df[PHAGE_COL].isin(test_phages[PHAGE_COL])].copy()

    return train_set, test_set


# ======================================
# PREPARAR FEATURES
# ======================================

def preparar_features(train_set, test_set):

    LEAK_VARS = [
        "NON-INFECTING PHAGES",
        "NON-INFECTABLE BACTERIA",
        "INFECTABLE BACTERIA",
        "NUMBER OF INFECTING PHAGES",
        "PHAGE",
        "BACTERIA"
    ]

    num_cols = train_set.select_dtypes(include=[float, int]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (["HOST RANGE", "AUC"] + LEAK_VARS)]

    cat_cols = train_set.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in (["HOST RANGE", "AUC"] + LEAK_VARS)]

    X_train = train_set[num_cols + cat_cols]
    X_test = test_set[num_cols + cat_cols]

    y_train = train_set["HOST RANGE"].astype(float)
    y_test = test_set["HOST RANGE"].astype(float)

    return X_train, X_test, y_train, y_test, num_cols, cat_cols


# ======================================
# ENTRENAR MODELO
# ======================================

def entrenar_modelo(train_set, test_set):

    X_train, X_test, y_train, y_test, num_cols, cat_cols = preparar_features(
        train_set,
        test_set
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_str", FunctionTransformer(lambda X: X.astype(str))),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=5000
    )

    pipeline = Pipeline(steps=[
        ("pre", preprocess),
        ("clf", clf)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


# ======================================
# PREDICCIÓN
# ======================================

def predecir_bacteria(pipeline, df, bacteria, threshold=0.6):

    df_bacteria = df[df["BACTERIA"] == bacteria].copy()

    X = df_bacteria.drop(columns=["HOST RANGE"], errors="ignore")

    probs = pipeline.predict_proba(X)[:, 1]

    df_bacteria["Probabilidad"] = probs
    df_bacteria["Prediccion"] = (probs >= threshold).astype(int)

    resultados = df_bacteria[
        ["PHAGE", "BACTERIA", "Probabilidad", "Prediccion"]
    ].sort_values("Probabilidad", ascending=False)

    return resultados
