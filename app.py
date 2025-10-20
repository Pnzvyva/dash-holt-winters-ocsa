# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tes_forecast import TESForecaster

def main():
    st.set_page_config(page_title="Pronóstico TES - Holt-Winters", layout="wide")

    # ---------------------------------------------------------------------
    # Encabezado con imágenes
    # ---------------------------------------------------------------------
    col1, col2 = st.columns([1, 4])

    with col1:
        st.image("images/OCSA-logo-fondo.png", width=180)

    with col2:
        st.image("images/banner_ocsa_un.png", use_container_width=True)

    st.markdown("---")

    st.title("Pronóstico de Tasas TES con Análisis de Diagnóstico")
    st.markdown(
        "Cargue un Excel con una hoja llamada 'Datos' (o se usará la primera) y una columna 'Fecha'."
    )

    if "forecaster" not in st.session_state:
        st.session_state.forecaster = TESForecaster()
    f = st.session_state.forecaster

    st.sidebar.header("Configuración de Datos")
    archivo = st.sidebar.file_uploader("Subir archivo Excel (.xlsx)", type=["xlsx"])

    if archivo is not None and st.sidebar.button("Cargar datos"):
        exito = f.load_data(archivo)
        if exito:
            st.sidebar.success("Datos cargados correctamente.")
        else:
            # getattr evita el AttributeError si por alguna razón no existiera
            st.sidebar.error(f"Error al cargar: {getattr(f, 'last_error', 'Error desconocido')}")

    if f.data is None:
        st.info("Aún no hay datos. Cargue un archivo para continuar.")
        return

    info = f.get_data_info()
    st.subheader("Información de los datos cargados")
    st.write(
        f"Período: {info['fecha_inicio'].strftime('%Y-%m-%d')} a {info['fecha_fin'].strftime('%Y-%m-%d')}"
    )
    st.write(f"Filas totales: {info['filas']}")
    st.dataframe(f.data.head())

    st.sidebar.header("Análisis y Pronóstico")
    columnas = f.get_numeric_columns()
    if not columnas:
        st.warning("No se encontraron columnas numéricas para análisis.")
        return

    columna = st.sidebar.selectbox("Columna a analizar", columnas)

    if st.sidebar.button("Ejecutar diagnóstico"):
        diag = f.run_diagnostics(columna)
        st.subheader(f"Resultados del diagnóstico: {columna}")
        for k, v in diag.items():
            st.write(f"{k}:", v)
        st.pyplot(f.plot_diagnostics(columna))

    st.sidebar.subheader("Parámetros Holt-Winters")
    periodos = st.sidebar.slider("Períodos de pronóstico", 7, 180, 30)
    estacionalidad = st.sidebar.slider("Longitud de estacionalidad", 2, 52, 12)
    tendencia = st.sidebar.selectbox("Tipo de tendencia", [None, "add", "mul"])
    tipo_estacional = st.sidebar.selectbox("Tipo de estacionalidad", [None, "add", "mul"])

    if st.sidebar.button("Generar pronóstico"):
        res = f.holt_winters_forecast(columna, periodos, estacionalidad, tendencia, tipo_estacional)

        st.subheader(f"Pronóstico para {columna}")
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=res["serie"].index, y=res["serie"].values, name="Serie histórica"))
        fig.add_trace(go.Scatter(x=res["ajuste"].index, y=res["ajuste"].values, name="Ajuste del modelo"))
        fechas_pron = pd.date_range(res["serie"].index[-1], periods=periodos + 1, freq="D")[1:]
        fig.add_trace(go.Scatter(x=fechas_pron, y=res["pronostico"].values, name="Pronóstico", line=dict(color="red")))
        fig.update_layout(title="Pronóstico Holt-Winters", xaxis_title="Fecha", yaxis_title="Valor")
        st.plotly_chart(fig, use_container_width=True)

        st.write("Métricas del modelo:")
        st.json(res["metricas"])

    # ---------------------------------------------------------------------
    # Pie de página con autoría
    # ---------------------------------------------------------------------
    mostrar_pie_de_pagina()


# -------------------------------------------------------------------------
# Función de pie de página
# -------------------------------------------------------------------------
def mostrar_pie_de_pagina():
    st.markdown("---")
    st.markdown("""
    Juan Camilo Pinedo Campo  
    Asistente de Investigación del Observatorio de Condiciones Socioeconómicas del Caribe Colombiano (OCSA)  
    **Universidad del Norte**
    """, unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:13px; color:gray;'>© 2025 OCSA - Todos los derechos reservados.</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
