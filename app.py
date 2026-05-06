import os
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
DATA_PATH = Path("insurance.csv")

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
NUMERICAS = ["age", "bmi", "children", "charges"]
CATEGORICAS = ["sex", "smoker", "region"]

MAPA_RIESGO_DEFAULT = {
    "0": "Riesgo bajo",
    "1": "Riesgo medio",
    "2": "Riesgo alto"
}


# =========================
# Entrenar modelo limpio
# =========================
def entrenar_modelo_desde_csv():
    if not DATA_PATH.exists():
        st.error(
            "No se pudo cargar el modelo .pkl y tampoco se encontró insurance.csv. "
            "Suba insurance.csv junto con app.py."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    faltantes = [col for col in FEATURES if col not in df.columns]
    if faltantes:
        st.error(f"insurance.csv no tiene estas columnas requeridas: {faltantes}")
        st.stop()

    X = df[FEATURES].copy()

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=10)),
        ]
    )

    modelo.fit(X)

    # Ordenar clusters por promedio de charges para asignar bajo, medio, alto.
    clusters = modelo.predict(X)
    resumen = (
        pd.DataFrame({"cluster": clusters, "charges": df["charges"]})
        .groupby("cluster")["charges"]
        .mean()
        .sort_values()
    )

    etiquetas = ["Riesgo bajo", "Riesgo medio", "Riesgo alto"]
    mapa_riesgo = {
        str(int(cluster)): etiqueta
        for cluster, etiqueta in zip(resumen.index, etiquetas)
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(modelo, MODEL_PATH, compress=3)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"mapa_riesgo": mapa_riesgo}, f, indent=4, ensure_ascii=False)

    return modelo, {"mapa_riesgo": mapa_riesgo}


# =========================
# Cargar modelo
# =========================
@st.cache_resource
def cargar_modelo_y_metadata():
    if MODEL_PATH.exists():
        try:
            modelo = joblib.load(MODEL_PATH, mmap_mode=None)

            if META_PATH.exists():
                with open(META_PATH, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = {"mapa_riesgo": MAPA_RIESGO_DEFAULT}

            return modelo, metadata

        except Exception:
            # Si el .pkl es incompatible, se reentrena automáticamente.
            st.warning(
                "El archivo .pkl no es compatible con este entorno. "
                "Se entrenará un modelo limpio usando insurance.csv."
            )
            return entrenar_modelo_desde_csv()

    return entrenar_modelo_desde_csv()


def obtener_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def explicar_con_gemini(cliente, cluster, riesgo):
    api_key = obtener_api_key()

    if not api_key:
        return "No se encontró GEMINI_API_KEY en Secrets."

    if genai is None:
        return "Falta instalar google-generativeai en requirements.txt."

    genai.configure(api_key=api_key)
    modelo_ia = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Explica este resultado para estudiantes de Gerencia Informática.

    Datos del cliente:
    {cliente}

    Cluster asignado: {cluster}
    Riesgo estimado: {riesgo}

    Redacta una explicación breve, didáctica y prudente.
    Aclara que es un ejercicio académico, no una decisión actuarial real.
    """

    respuesta = modelo_ia.generate_content(prompt)
    return respuesta.text


# =========================
# Aplicación
# =========================
modelo, metadata = cargar_modelo_y_metadata()
mapa_riesgo = metadata.get("mapa_riesgo", MAPA_RIESGO_DEFAULT)

st.title("Clasificación de riesgo actuarial")
st.write(
    "Aplicación en Streamlit para usar un modelo K-Means sobre datos de seguros médicos."
)

with st.sidebar:
    st.header("Datos del cliente")

    age = st.slider("Edad", 18, 70, 45)
    sex = st.selectbox("Sexo", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 31.2, step=0.1)
    children = st.slider("Hijos", 0, 5, 2)
    smoker = st.selectbox("Fumador", ["yes", "no"])
    region = st.selectbox("Región", ["southeast", "southwest", "northeast", "northwest"])
    charges = st.number_input("Cargos médicos", 0.0, 70000.0, 28000.0, step=500.0)

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

st.divider()
st.caption(
    "Archivos recomendados: app.py, requirements.txt, insurance.csv y carpeta models. "
    "Si el .pkl falla, la app reentrena el modelo desde insurance.csv."
)
