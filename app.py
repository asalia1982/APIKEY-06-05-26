import os
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# IA opcional para explicar el resultado.
# En requirements.txt agregue: google-generativeai
try:
    import google.generativeai as genai
except Exception:
    genai = None


# =========================
# Configuración general
# =========================
st.set_page_config(
    page_title="Riesgo actuarial con K-Means",
    layout="centered"
)

MODEL_PATH = Path("models/kmeans_riesgo_actuarial.pkl")
META_PATH = Path("models/model_metadata.json")


# =========================
# Funciones auxiliares
# =========================
@st.cache_resource
def cargar_modelo():
    if not MODEL_PATH.exists():
        st.error(f"No se encontró el modelo en: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_metadata():
    if not META_PATH.exists():
        return {"mapa_riesgo": {}}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def obtener_api_key():
    # Recomendado para Streamlit Cloud:
    # En Settings > Secrets agregue:
    # GEMINI_API_KEY = "su_clave"
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def interpretar_con_gemini(datos_cliente, cluster, riesgo, api_key):
    if not api_key or genai is None:
        return None

    genai.configure(api_key=api_key)
    modelo_ia = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Actúa como asistente académico de Gerencia Informática.
    Explica de forma breve, clara y didáctica el resultado de un modelo
    de segmentación de riesgo actuarial.

    Datos del cliente:
    {datos_cliente}

    Cluster asignado: {cluster}
    Nivel de riesgo estimado: {riesgo}

    Incluye:
    1. Interpretación del perfil.
    2. Posible explicación del nivel de riesgo.
    3. Advertencia de que es un ejercicio académico, no una decisión financiera real.
    """

    respuesta = modelo_ia.generate_content(prompt)
    return respuesta.text


# =========================
# Carga del modelo
# =========================
modelo = cargar_modelo()
metadata = cargar_metadata()
mapa_riesgo = metadata.get("mapa_riesgo", {})


# =========================
# Interfaz
# =========================
st.title("Modelo de riesgo actuarial")
st.write(
    "Aplicación didáctica para clasificar un cliente usando el modelo K-Means "
    "entrenado en clase."
)

with st.sidebar:
    st.header("Datos del cliente")

    age = st.slider("Edad", 18, 70, 45)
    sex = st.selectbox("Sexo", ["male", "female"])
    bmi = st.number_input("Índice de masa corporal BMI", 10.0, 60.0, 31.2, step=0.1)
    children = st.slider("Número de hijos", 0, 5, 2)
    smoker = st.selectbox("Fumador", ["yes", "no"])
    region = st.selectbox(
        "Región",
        ["southeast", "southwest", "northeast", "northwest"]
    )
    charges = st.number_input("Cargos médicos", 0.0, 70000.0, 28000.0, step=500.0)

    usar_ia = st.checkbox("Generar interpretación con Gemini", value=False)


cliente = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
    "charges": charges
}])


st.subheader("Datos ingresados")
st.dataframe(cliente, use_container_width=True)


if st.button("Predecir riesgo"):
    cluster = int(modelo.predict(cliente)[0])
    riesgo = mapa_riesgo.get(str(cluster), f"Cluster {cluster}")

    st.subheader("Resultado del modelo")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Cluster asignado", cluster)

    with col2:
        st.metric("Nivel de riesgo", riesgo)

    st.info(
        "Este resultado proviene de un modelo de clustering K-Means. "
        "Debe interpretarse como práctica académica, no como recomendación actuarial real."
    )

    if usar_ia:
        api_key = obtener_api_key()

        if not api_key:
            st.warning(
                "No se encontró GEMINI_API_KEY. Configure la clave en los Secrets "
                "de Streamlit o como variable de entorno."
            )
        elif genai is None:
            st.warning(
                "Falta instalar google-generativeai. Agréguelo al archivo requirements.txt."
            )
        else:
            with st.spinner("Generando interpretación con Gemini..."):
                try:
                    explicacion = interpretar_con_gemini(
                        cliente.to_dict(orient="records")[0],
                        cluster,
                        riesgo,
                        api_key
                    )
                    st.subheader("Interpretación con IA")
                    st.write(explicacion)
                except Exception as e:
                    st.error(f"No se pudo generar la interpretación con Gemini: {e}")


st.divider()
st.caption(
    "Archivos esperados: models/kmeans_riesgo_actuarial.pkl y "
    "models/model_metadata.json"
)
