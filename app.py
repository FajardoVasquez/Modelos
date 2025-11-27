import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuración de la página
st.set_page_config(
    page_title="ML App - Social Media & Mental Health",
    page_icon="🧠",
    layout="wide"
)

# Título principal
st.title("🧠 Análisis de Salud Mental y Redes Sociales")
st.markdown("**Modelos:** Árbol de Decisión (Gini) + PCA")

# Sidebar para navegación
st.sidebar.title("⚙️ Configuración")
modo = st.sidebar.radio(
    "Selecciona el modo:",
    ["🏠 Inicio", "📊 Modo Supervisado", "🔍 Modo No Supervisado", "💾 Exportación"]
)

# ========== CARGA DEL DATASET DESDE ARCHIVO LOCAL ==========

@st.cache_data
def cargar_datos():
    try:
        with st.spinner("Cargando dataset local..."):
            df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")  # Ajusta el nombre si es diferente
            return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'mental_health.csv' en /data/")
        st.info("Verifica que la ruta correcta sea:  /data/mental_health.csv")
        return None
    except Exception as e:
        st.error(f"Error al cargar el dataset: {str(e)}")
        return None

# Cargar dataset
df_original = cargar_datos()
if df_original is None:
    st.stop()

# Preprocesamiento
@st.cache_data
def preprocesar_datos(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    df = df.fillna(df.mean(numeric_only=True))

    return df, label_encoders, numeric_cols, categorical_cols

df, label_encoders, numeric_cols, categorical_cols = preprocesar_datos(df_original)

# Variable objetivo
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Variable Objetivo")

target_column = st.sidebar.selectbox(
    "Selecciona la variable a predecir:",
    df.columns.tolist()
)

# Preparar X e y
X = df.drop(columns=[target_column]).values
y = df[target_column].values
feature_names = df.drop(columns=[target_column]).columns.tolist()
target_names = np.unique(y)

# Estado de sesión
if 'modelo_supervisado' not in st.session_state:
    st.session_state.modelo_supervisado = None
if 'modelo_no_supervisado' not in st.session_state:
    st.session_state.modelo_no_supervisado = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'metricas_supervisado' not in st.session_state:
    st.session_state.metricas_supervisado = {}
if 'metricas_no_supervisado' not in st.session_state:
    st.session_state.metricas_no_supervisado = {}
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

# ================== MODO INICIO ==================
if modo == "🏠 Inicio":
    st.header("📊 Bienvenido al Análisis de Salud Mental y Redes Sociales")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Modelo Supervisado")
        st.info("**Árbol de Decisión (Gini)**")

    with col2:
        st.subheader("🔍 Modelo No Supervisado")
        st.info("**PCA - Reducción de Dimensionalidad**")

    st.markdown("---")
    st.subheader("📈 Vista Previa del Dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Registros", f"{len(df):,}")
    col2.metric("Características", len(feature_names))
    col3.metric("Clases Únicas", len(target_names))
    col4.metric("Variable Objetivo", target_column)

    st.dataframe(df_original.head(10), use_container_width=True)

    st.subheader("📊 Distribución de la Variable Objetivo")
    fig = px.histogram(df_original, x=target_column)
    st.plotly_chart(fig, use_container_width=True)

# ================== MODO SUPERVISADO ==================
elif modo == "📊 Modo Supervisado":
    st.header("📊 Árbol de Decisión (Criterio Gini)")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_depth = st.slider("Profundidad Máxima", 1, 20, 5)
    with col2:
        min_samples_split = st.slider("Mínimo Muestras para Dividir", 2, 20, 2)
    with col3:
        min_samples_leaf = st.slider("Mínimo Muestras en Hoja", 1, 10, 1)

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)

    if st.button("🚀 Entrenar Modelo Supervisado", type="primary"):
        with st.spinner("Entrenando modelo..."):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )

                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                modelo = DecisionTreeClassifier(
                    criterion='gini',
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
                modelo.fit(X_train, y_train)

                y_pred = modelo.predict(X_test)
                st.session_state.y_pred = y_pred

                metricas = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average='weighted')),
                    'recall': float(recall_score(y_test, y_pred, average='weighted')),
                    'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
                }

                st.session_state.modelo_supervisado = modelo
                st.session_state.metricas_supervisado = metricas

                st.success("✅ Modelo entrenado exitosamente!")

            except Exception as e:
                st.error(f"Error al entrenar el modelo: {str(e)}")

    # Mostrar métricas
    if st.session_state.modelo_supervisado:
        st.markdown("---")
        st.subheader("📈 Métricas del Modelo")

        col1, col2, col3, col4 = st.columns(4)
        m = st.session_state.metricas_supervisado

        col1.metric("Accuracy", f"{m['accuracy']:.4f}")
        col2.metric("Precision", f"{m['precision']:.4f}")
        col3.metric("Recall", f"{m['recall']:.4f}")
        col4.metric("F1-Score", f"{m['f1_score']:.4f}")

# ================== MODO NO SUPERVISADO ==================
elif modo == "🔍 Modo No Supervisado":
    st.header("🔍 PCA - Análisis de Componentes Principales")

    max_components = min(len(feature_names), 20)
    n_components = st.slider("Número de Componentes", 2, max_components, 3)
    n_clusters = st.slider("Clusters (KMeans)", 2, 10, 3)

    if st.button("🚀 Aplicar PCA y Clustering", type="primary"):
        with st.spinner("Aplicando PCA..."):
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)

                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_pca)

                metricas = {
                    'silhouette_score': float(silhouette_score(X_pca, cluster_labels)),
                    'davies_bouldin': float(davies_bouldin_score(X_pca, cluster_labels)),
                    'variance_explained': pca.explained_variance_ratio_.tolist(),
                    'total_variance': float(sum(pca.explained_variance_ratio_))
                }

                st.session_state.modelo_no_supervisado = pca
                st.session_state.metricas_no_supervisado = metricas
                st.session_state.X_pca = X_pca
                st.session_state.cluster_labels = cluster_labels

                st.success("✅ PCA aplicado!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ================== EXPORTACIÓN ==================
elif modo == "💾 Exportación":
    st.header("💾 Exportación de Modelos")

    col1, col2 = st.columns(2)

    # Supervisado
    with col1:
        st.subheader("📊 Modelo Supervisado")
        if st.session_state.modelo_supervisado:
            modelo = st.session_state.modelo_supervisado

            json_data = {
                "model_type": "Supervised",
                "model_name": "Decision Tree (Gini)",
                "target_variable": target_column,
                "metrics": st.session_state.metricas_supervisado,
                "features": feature_names,
                "feature_importances": modelo.feature_importances_.tolist(),
            }

            st.download_button(
                "📥 Descargar JSON",
                data=json.dumps(json_data, indent=2),
                file_name="modelo_supervisado.json"
            )

            st.download_button(
                "📥 Descargar Modelo (.pkl)",
                data=pickle.dumps(modelo),
                file_name="modelo_supervisado.pkl"
            )

    # No supervisado
    with col2:
        st.subheader("🔍 Modelo No Supervisado")
        if st.session_state.modelo_no_supervisado:
            json_data = {
                "model_type": "Unsupervised",
                "algorithm": "PCA",
                "metrics": st.session_state.metricas_no_supervisado,
            }

            st.download_button(
                "📥 Descargar JSON PCA",
                data=json.dumps(json_data, indent=2),
                file_name="modelo_pca.json"
            )
