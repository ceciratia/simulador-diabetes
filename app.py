"""Simulador de riesgo de diabetes silenciosa - TFM "El coste del silencio"."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cribado de diabetes silenciosa",
                   page_icon="\U0001FA7A", layout="wide")

@st.cache_resource
def cargar():
    return (joblib.load("random_forest.pkl"), joblib.load("segmentador.pkl"),
            joblib.load("modelo_coste.pkl"), joblib.load("contexto.pkl"))

modelo, segmentador, coste, ctx = cargar()
CONSULTA, VARS_SEG = ctx["consulta"], segmentador["variables"]
UMBRAL = 0.45

ARQ = {0: ("Riesgo temprano",
           "Intervencion preventiva sobre estilo de vida. Maximo recorrido."),
       1: ("Riesgo por edad",
           "Control y seguimiento. Vigilar acumulacion de comorbilidades."),
       2: ("Hipertenso oculto",
           "Prioridad alta: control tensional y analitica sin demora."),
       3: ("Obesidad severa temprana",
           "Programa de estilo de vida con apoyo psicologico asociado.")}

st.title("Cribado de diabetes silenciosa")
st.caption("Estimacion de riesgo, arquetipo clinico y coste sanitario "
           "proyectado a partir de datos de consulta ordinaria. "
           "Sin analitica previa.")

with st.sidebar:
    st.header("Datos del paciente")
    edad = st.slider("Edad", 20, 80, 55)
    sexo = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
    st.divider()
    peso = st.number_input("Peso (kg)", 40.0, 200.0, 82.0, 0.5)
    talla = st.number_input("Talla (cm)", 140.0, 210.0, 172.0, 0.5)
    cintura = st.number_input("Perimetro de cintura (cm)", 50.0, 180.0, 98.0, 0.5)
    st.caption("El perimetro de cintura es la variable mas predictiva "
               "del modelo, por encima del IMC.")
    st.divider()
    sis = st.number_input("Tension sistolica", 80, 220, 128)
    dia = st.number_input("Tension diastolica", 40, 140, 78)
    st.divider()
    hta = st.checkbox("Hipertension diagnosticada")
    col = st.checkbox("Colesterol alto diagnosticado")
    cor = st.checkbox("Enfermedad coronaria")
    inf = st.checkbox("Infarto previo")
    ict = st.checkbox("Ictus previo")
    st.divider()
    sedent = st.slider("Minutos sedentarios al dia", 0, 900, 360, 30)
    sueno = st.slider("Horas de sueno", 3, 12, 7)
    phq = st.slider("Puntuacion PHQ-9 (depresion)", 0, 27, 3)
    tabaco = st.selectbox("Tabaquismo", ["Nunca", "Exfumador", "Fumador"])
    alcohol = st.slider("Frecuencia de alcohol (0-10)", 0, 10, 2)
    cambio = st.number_input("Cambio de peso en el ultimo ano (kg)",
                             -40.0, 40.0, 0.0, 0.5)

imc = peso / (talla / 100) ** 2
datos = {"RIDAGEYR": edad, "RIAGENDR": 1 if sexo == "Hombre" else 2,
         "BMXWT": peso, "BMXHT": talla, "BMXWAIST": cintura, "BMXBMI": imc,
         "ratio_cintura_talla": cintura / talla,
         "ta_sistolica": sis, "ta_diastolica": dia,
         "BPQ020": 1 if hta else 2, "BPQ080": 1 if col else 2,
         "MCQ160C": 1 if cor else 2, "MCQ160E": 1 if inf else 2,
         "MCQ160F": 1 if ict else 2, "MCQ160B": 2, "MCQ160L": 2,
         "PAD680": sedent, "SLD012": sueno, "phq9": phq,
         "estado_tabaquico": ["Nunca", "Exfumador", "Fumador"].index(tabaco),
         "frecuencia_alcohol": alcohol,
         "cambio_peso_anual": cambio * 2.205}

fila = pd.DataFrame([datos])
for c in CONSULTA:
    if c not in fila.columns:
        fila[c] = np.nan

probs = modelo.predict_proba(fila[CONSULTA])[0]
p_riesgo = probs[1] + probs[2]

n_com = sum([hta, col, cor, inf, ict])
datos["n_comorbilidades"] = n_com
seg = pd.DataFrame([{v: datos.get(v, np.nan) for v in VARS_SEG}])
arq = int(segmentador["kmeans"].predict(segmentador["prep"].transform(seg))[0])

e = (edad - 50) / 10
x = pd.DataFrame([{"const": 1.0, "edad_c": e, "edad_c2": e ** 2,
                   "mujer": 1 if sexo == "Mujer" else 0,
                   "diabetes": 1 if probs[2] > 0.5 else 0,
                   "n_comorbilidades": n_com}])
coste_anual = float(coste["modelo"].predict(x).iloc[0])

c1, c2, c3 = st.columns(3)
c1.metric("Probabilidad de riesgo", f"{p_riesgo:.0%}")
c2.metric("Probabilidad de diabetes", f"{probs[2]:.0%}")
c3.metric("Coste anual proyectado", f"{coste_anual:,.0f} $")

if p_riesgo >= UMBRAL:
    st.error(f"**Priorizar para cribado.** Riesgo estimado del {p_riesgo:.0%}, "
             f"por encima del umbral operativo del {UMBRAL:.0%}.")
else:
    st.success(f"**No priorizar en esta ronda.** Riesgo estimado del "
               f"{p_riesgo:.0%}, por debajo del umbral operativo.")

st.subheader(f"Arquetipo: {ARQ[arq][0]}")
st.write(ARQ[arq][1])

with st.expander("Como leer este resultado"):
    st.markdown(f"""
El umbral operativo es del **{UMBRAL:.0%}**, no del 50 %. En cribado
poblacional un falso negativo —un paciente devuelto a casa— cuesta ordenes
de magnitud mas que un falso positivo —una analitica innecesaria—, de modo
que el punto de corte se desplaza deliberadamente hacia la deteccion.

**IMC calculado:** {imc:.1f} | **Ratio cintura/talla:** {cintura/talla:.2f}

El ratio cintura/talla es la variable de mayor peso en el modelo, por delante
de la edad y muy por delante del IMC. Un paciente con IMC normal y cintura
elevada puede presentar riesgo alto: es el perfil que los criterios de cribado
vigentes no detectan.

El coste proyectado es una estimacion basada en gasto sanitario
estadounidense (MEPS 2024) y no es trasladable directamente a otro sistema.
    """)

st.caption("TFM 'El coste del silencio'. Herramienta de apoyo a la decision "
           "sobre priorizacion de cribado. No constituye diagnostico medico.")