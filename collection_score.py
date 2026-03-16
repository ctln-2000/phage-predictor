import pandas as pd
import numpy as np

def calcular_collection_score(
    fagos_df,
    df_original,
    df_polinomio,
    usar_probabilidad=True,
    prob_col="Probabilidad",
    hr_col="Prediccion"
):

    df = fagos_df.copy()

    # =============================
    # MERGE CON DATOS ORIGINALES
    # =============================

    df = df.merge(
        df_original[
            [
                "PHAGE","BACTERIA","LIFE CYCLE","AUC",
                "Tasa de Eclosión","Periodo de latencia",
                "T° SCORE","pH SCORE","UV SCORE"
            ]
        ].drop_duplicates(),
        on=["PHAGE","BACTERIA"],
        how="left"
    )

    df = df.merge(
        df_polinomio[["PHAGE","HR2","infR","FR","PROD"]].drop_duplicates(),
        on="PHAGE",
        how="left"
    )

    # =============================
    # S1 SEGURIDAD
    # =============================

    df["Lc"] = df["LIFE CYCLE"].apply(
        lambda x: 1 if str(x).lower() in ["lytic","litico","lítico","1"] else 0
    )

    if usar_probabilidad:
        df["S1"] = df["Lc"] * df[prob_col]
    else:
        df["S1"] = df["Lc"] * df[hr_col]

    # =============================
    # S2 INFECTIVIDAD
    # =============================

    df["AUC"] = df["AUC"].fillna(0).clip(lower=0)

    med_bs = df_original["Tasa de Eclosión"].median()
    med_lat = df_original["Periodo de latencia"].median()

    df["Tasa de Eclosión"] = df["Tasa de Eclosión"].fillna(med_bs)
    df["Periodo de latencia"] = df["Periodo de latencia"].replace(0, med_lat)

    df["BS_Lat_ratio"] = (df["Tasa de Eclosión"] / df["Periodo de latencia"]) ** 1.5

    df_temp = df_original.copy()
    df_temp["Periodo de latencia"] = df_temp["Periodo de latencia"].replace(0, med_lat)

    max_ratio = ((df_temp["Tasa de Eclosión"] / df_temp["Periodo de latencia"]) ** 1.5).max()

    df["BS_Lat_ratio_norm"] = df["BS_Lat_ratio"] / max_ratio if max_ratio > 0 else 0

    df["HR2"] = df["HR2"].fillna(0)

    df["S2"] = (
        0.6 * df["AUC"]
        + 0.25 * df["BS_Lat_ratio_norm"]
        + 0.15 * df["HR2"]
    )

    # =============================
    # S3 RESISTENCIA
    # =============================

    df["infR"] = df["infR"].fillna(0)
    df["FR"] = df["FR"].fillna(0)

    df["S3"] = df["infR"] / (df["FR"] + df["infR"] + 1e-10)

    # =============================
    # S4 VIABILIDAD
    # =============================

    df["PROD"] = df["PROD"].fillna(0)

    for col in ["T° SCORE","pH SCORE","UV SCORE"]:
        df[col] = df[col].fillna(0)
        if df[col].max() > 1:
            df[col] = df[col] / df[col].max()

    df["S4"] = df["PROD"] * (
        (df["T° SCORE"] + df["pH SCORE"] + df["UV SCORE"]) / 3
    )

    # =============================
    # COLLECTION SCORE
    # =============================

    df["CS_cocktail"] = df["S1"] * (
        (0.6 * df["S2"])
        + (0.3 * df["S3"])
        + (0.1 * df["S4"])
    )

    df["CS_cocktail"] = df["CS_cocktail"].clip(0,1)

    return df.sort_values("CS_cocktail", ascending=False)
