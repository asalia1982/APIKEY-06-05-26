import os
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

try:
    import google.generativeai as genai
except Exception:
    genai = None


# =========================
# Configuración
# =========================
st.set_page_config(
    page_title="Riesgo actuarial",
    layout="centered"
)

MODEL_PATH = Path("models/kmeans_riesgo_actuarial.pkl")
META_PATH = Path("models/model_metadata.json")


# =========================
# Cargar modelo y metadata
# =========================
@st.cache_resource
def cargar_modelo():
    if not MODEL_PATH.exists():
        st.error("No se encontró el archivo models/kmeans_riesgo_actuarial.pkl")
        st.stop()

    try:
        return joblib.load(MODEL_PATH, mmap_mode=None)
    except Exception as e:
        st.error("No se pudo cargar el modelo .pkl")
        st.code(str(e))
        st.warning(
            "Revise que el modelo haya sido guardado con una versión compatible "
            "de Python y scikit-learn."
        )
        st.stop()


@st.cache_data
def cargar_metadata():
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "mapa_riesgo": {
            "0": "Riesgo bajo",
            "1": "Riesgo medio",
            "2": "Riesgo alto"
        }
    }


def obtener_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def explicar_con_gemini(cliente, cluster, riesgo):
    api_key = obtener_api_key()

    if not api_key:
        return "No se encontró GEMINI_API_KEY en los Secrets de Streamlit."

    if genai is None:
        return "Falta instalar google-generativeai en requirements.txt."

    genai.configure(api_key=api_key)
    modelo_ia = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Explica brevemente este resultado de clustering actuarial para estudiantes.

    Datos del cliente:
    {cliente}

    Cluster asignado: {cluster}
    Riesgo estimado: {riesgo}

    Redacta:
    - Interpretación del perfil.
    - Razón probable del riesgo.
    - Advertencia de que es un ejercicio académico.
    """

    respuesta = modelo_ia.generate_content(prompt)
    return respuesta.text


# =========================
# App
# =========================
modelo = cargar_modelo()
metadata = cargar_metadata()
mapa_riesgo = metadata.get("mapa_riesgo", {})

st.title("Clasificación de riesgo actuarial")
st.write("Aplicación en Streamlit para usar el modelo K-Means entrenado en clase.")

with st.sidebar:
    st.header("Datos del cliente")

    age = st.slider("Edad", 18, 70, 45)
    sex = st.selectbox("Sexo", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=31.2, step=0.1)
    children = st.slider("Hijos", 0, 5, 2)
    smoker = st.selectbox("Fumador", ["yes", "no"])
    region = st.selectbox("Región", ["southeast", "southwest", "northeast", "northwest"])
    charges = st.number_input("Cargos médicos", min_value=0.0, max_value=70000.0, value=28000.0, step=500.0)

    usar_ia = st.checkbox("Explicar resultado con Gemini")

cliente = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
    "charges": charges
}])

st.subheader("Cliente ingresado")
st.dataframe(cliente, use_container_width=True)

if st.button("Predecir"):
    try:
        cluster = int(modelo.predict(cliente)[0])
        riesgo = mapa_riesgo.get(str(cluster), f"Cluster {cluster}")

        st.subheader("Resultado")
        col1, col2 = st.columns(2)

        col1.metric("Cluster", cluster)
        col2.metric("Riesgo estimado", riesgo)

        st.info(
            "Este resultado es académico. No debe usarse como decisión financiera, "
            "médica o actuarial real."
        )

        if usar_ia:
            with st.spinner("Generando explicación con Gemini..."):
                explicacion = explicar_con_gemini(
                    cliente.to_dict(orient="records")[0],
                    cluster,
                    riesgo
                )
                st.subheader("Explicación con IA")
                st.write(explicacion)

    except Exception as e:
        st.error("No se pudo realizar la predicción.")
        st.code(str(e))
        st.warning(
            "Verifique que las columnas del cliente coincidan con las columnas "
            "usadas al entrenar el modelo."
        )

st.divider()
st.caption("Archivos requeridos: app.py, requirements.txt, models/kmeans_riesgo_actuarial.pkl y models/model_metadata.json")
