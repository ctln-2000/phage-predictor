import streamlit as st
import pandas as pd
import io

from modelo import crear_train_test, entrenar_modelo, predecir_bacteria
from collection_score import calcular_collection_score
from graficos import heatmap_cs

st.title("🧬 Predicción Fago-Bacteria")

# ======================================
# CARGAR DATOS
# ======================================


def cargar():

    # =========================
    # TABLA PRINCIPAL
    # =========================

    df = pd.read_excel("TABLA_FINAL_RELLENA.xlsx")

    df["Tasa de Eclosión"] = df["Tasa de Eclosión"].replace(
        ["ND","nd","Nd","Not Available","Not Aviable","not aviable"],0
    )

    df["Periodo de latencia"] = df["Periodo de latencia"].replace(
        ["ND","nd","Nd","Not Available","Not Aviable","not aviable"],60
    )

    df = df[df["BACTERIA"] != "PsCx689"].copy()


    # =========================
    # TABLA POLINOMIO
    # =========================

    df_polinomio = pd.read_excel("polinomio_datos.xlsx")


    # =========================
    # ENTRENAR MODELO
    # =========================

    train_set, test_set = crear_train_test(df)

    pipeline, X_test, y_test = entrenar_modelo(train_set, test_set)

    return df, df_polinomio, pipeline


df, df_polinomio, modelo = cargar()

st.success("Modelo entrenado")


def convertir_excel(df):

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")

    return output.getvalue()


# ======================================
# SELECCIONAR BACTERIA
# ======================================

bacterias = sorted(df["BACTERIA"].unique())

bacteria = st.selectbox(
    "Selecciona bacteria",
    bacterias
)

# ======================================
# PREDICCIÓN
# ======================================

# ======================================
# PREDICCIÓN
# ======================================

if st.button("Predecir fagos"):

    resultados = predecir_bacteria(
        modelo,
        df,
        bacteria
    )

    # ============================
    # CALCULAR COLLECTION SCORE
    # ============================

    resultados = calcular_collection_score(
        resultados,
        df,
        df_polinomio,
        prob_col="Probabilidad"
    )
    # ordenar columnas
    cols = resultados.columns.tolist()

    if "CS_cocktail" in cols and "Probabilidad" in cols:
        cols.insert(cols.index("Probabilidad") + 1, cols.pop(cols.index("CS_cocktail")))

    resultados = resultados[cols]

    # ============================
    # MOSTRAR TABLA
    # ============================

    st.subheader("Ranking de Fagos")

    st.dataframe(resultados)

    excel = convertir_excel(resultados)

    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=excel,
        file_name="ranking_fagos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ============================
    # GRAFICO PROBABILIDAD
    # ============================

    st.bar_chart(
        resultados.set_index("PHAGE")["Probabilidad"]
    )

    # ============================
    # HEATMAP CS
    # ============================

    fig = heatmap_cs(resultados)

    st.pyplot(fig)
