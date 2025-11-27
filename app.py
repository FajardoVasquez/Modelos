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

# ---------------------
# Configuración de la página
# ---------------------
st.set_page_config(
    page_title="ML App - Social Media & Mental Health",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Análisis de Salud Mental y Redes Sociales")
st.markdown("**Modelos:** Árbol de Decisión (Gini) + PCA")

# ---------------------
# Sidebar - uploader
# ---------------------
st.sidebar.title("⚙️ Configuración")
st.sidebar.subheader("📂 Carga del Dataset")

uploaded_file = st.sidebar.file_uploader("📁 Subir dataset (CSV)", type=["csv"])

@st.cache_data
def cargar_desde_upload(archivo):
    try:
        df = pd.read_csv(archivo)
        return df
    except Exception as e:
        st.error(f"❌ Error leyendo el archivo cargado: {e}")
        return None

# Cargar dataset (solo por uploader, según tu petición)
if uploaded_file is None:
    st.sidebar.info("Sube un CSV para comenzar.")
    st.stop()
else:
    df_original = cargar_desde_upload(uploaded_file)
    if df_original is None:
        st.stop()

# ---------------------
# Preprocesamiento
# ---------------------
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

    # Rellenar valores numéricos faltantes
    df = df.fillna(df.mean(numeric_only=True))

    return df, label_encoders, numeric_cols, categorical_cols

df, label_encoders, numeric_cols, categorical_cols = preprocesar_datos(df_original)

# ---------------------
# Variable objetivo
# ---------------------
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

# ---------------------
# Session state
# ---------------------
if 'modelo_supervisado' not in st.session_state:
    st.session_state.modelo_supervisado = None
if 'modelo_no_supervisado' not in st.session_state:
    st.session_state.modelo_no_supervisado = None
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
if 'X_pca' not in st.session_state:
    st.session_state.X_pca = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None

# ---------------------
# Menú principal
# ---------------------
modo = st.sidebar.radio(
    "Selecciona el modo:",
    ["🏠 Inicio", "📊 Modo Supervisado", "🔍 Modo No Supervisado", "💾 Exportación"]
)

# ================== MODO INICIO ==================
if modo == "🏠 Inicio":
    st.header("📊 Bienvenido al Análisis de Salud Mental y Redes Sociales")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Registros", f"{len(df):,}")
    col2.metric("Características", len(feature_names))
    col3.metric("Clases Únicas", len(target_names))
    col4.metric("Variable Objetivo", target_column)

    st.subheader("Vista previa del dataset")
    st.dataframe(df_original.head(10), use_container_width=True)

    st.subheader("Descripción rápida")
    st.write(df_original.describe(include='all').transpose())

# ================== MODO SUPERVISADO ==================
elif modo == "📊 Modo Supervisado":
    st.header("📊 Árbol de Decisión (Criterio Gini)")

    # Hiperparámetros
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

    # Entrenamiento
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
                    'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                }

                st.session_state.modelo_supervisado = modelo
                st.session_state.metricas_supervisado = metricas

                st.success("✅ Modelo entrenado exitosamente!")

            except Exception as e:
                st.error(f"Error al entrenar el modelo: {str(e)}")

    # Mostrar métricas y gráficas si está entrenado
    if st.session_state.modelo_supervisado:
        st.markdown("---")
        st.subheader("📈 Métricas del Modelo")

        m = st.session_state.metricas_supervisado
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{m['accuracy']:.4f}")
        col2.metric("Precision", f"{m['precision']:.4f}")
        col3.metric("Recall", f"{m['recall']:.4f}")
        col4.metric("F1-Score", f"{m['f1_score']:.4f}")

        # Gráfica 1: Importancia de características (bar chart)
        try:
            modelo = st.session_state.modelo_supervisado
            importances = modelo.feature_importances_
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False)

            st.subheader("📊 Importancia de características")
            fig_fi = px.bar(fi_df, x="feature", y="importance", title="Feature importances (Decision Tree)")
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar gráfico de importancias: {e}")

        # Gráfica 2: Matriz de confusión (heatmap)
        try:
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=[str(x) for x in target_names], columns=[str(x) for x in target_names])

            st.subheader("🔢 Matriz de Confusión")
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_df.values,
                x=cm_df.columns,
                y=cm_df.index,
                hoverongaps=False,
                showscale=True
            ))
            fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar la matriz de confusión: {e}")

        # Sección de prueba interactiva: sliders para una predicción manual
        st.markdown("---")
        st.subheader("🧪 Prueba Interactiva - Predicción Manual")

        # Construir controles dinámicos para cada feature (si son numéricos)
        manual_input = {}
        st.write("Introduce valores para hacer una predicción (si hay muchas características, usa valores razonables).")
        for feat in feature_names:
            # Si es numérico aproximamos con min-max del dataframe preprocesado
            col_values = df[feat]
            if pd.api.types.is_numeric_dtype(col_values):
                min_v = float(np.nanmin(col_values))
                max_v = float(np.nanmax(col_values))
                mean_v = float(np.nanmean(col_values))
                val = st.number_input(f"{feat}", min_value=min_v, max_value=max_v, value=mean_v)
                manual_input[feat] = val
            else:
                # Si no es numérico (ya fue label-encoded), usar selectbox con valores únicos
                uniques = sorted(list(np.unique(col_values)))
                val = st.selectbox(f"{feat}", uniques, index=0)
                manual_input[feat] = val

        if st.button("🔮 Predecir con el Árbol"):
            try:
                input_arr = np.array([manual_input[f] for f in feature_names]).reshape(1, -1)
                pred = st.session_state.modelo_supervisado.predict(input_arr)[0]
                st.success(f"Predicción: {pred} (label: {label_encoders.get(target_column).inverse_transform([int(pred)])[0] if target_column in label_encoders else pred})")
                # Añadir JSON de la predicción para descargar
                pred_json = {
                    "input": manual_input,
                    "output_class": int(pred) if np.issubdtype(type(pred), np.integer) or np.issubdtype(np.array(pred).dtype, np.integer) else pred,
                }
                st.json(pred_json)
            except Exception as e:
                st.error(f"No se pudo realizar la predicción manual: {e}")

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

    # Mostrar métricas y gráficas si PCA está aplicado
    if st.session_state.modelo_no_supervisado:
        st.markdown("---")
        st.subheader("📈 Métricas del PCA + Clustering")
        m = st.session_state.metricas_no_supervisado

        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{m['silhouette_score']:.4f}")
        col2.metric("Davies-Bouldin", f"{m['davies_bouldin']:.4f}")
        col3.metric("Varianza total (explicada)", f"{m['total_variance']:.4f}")

        # Gráfica 1: Scree plot (varianza explicada por componente)
        try:
            var_explained = m['variance_explained']
            scree_df = pd.DataFrame({
                "component": [f"PC{i+1}" for i in range(len(var_explained))],
                "variance": var_explained
            })
            st.subheader("📈 Scree plot (Varianza explicada por componente)")
            fig_scree = px.bar(scree_df, x="component", y="variance", title="Explained variance ratio (PCA)")
            st.plotly_chart(fig_scree, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar scree plot: {e}")

        # Gráfica 2: Scatter de PC1 vs PC2 coloreado por cluster
        try:
            X_pca = st.session_state.X_pca
            labels = st.session_state.cluster_labels
            # construir dataframe para plotly
            pca_df = pd.DataFrame({
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1] if X_pca.shape[1] > 1 else np.zeros(X_pca.shape[0]),
                "cluster": labels.astype(str)
            })
            st.subheader("🔎 Scatter plot (PC1 vs PC2)")
            fig_scatter = px.scatter(pca_df, x="PC1", y="PC2", color="cluster", title="PCA scatter by cluster")
            st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar el scatter plot: {e}")

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
        else:
            st.info("Entrena primero el modelo supervisado para exportarlo.")

    # No supervisado
    with col2:
        st.subheader("🔍 Modelo No Supervisado")
        if st.session_state.modelo_no_supervisado:
            json_data = {
                "model_type": "Unsupervised",
                "algorithm": "PCA",
                "metrics": st.session_state.metricas_no_supervisado,
                "cluster_labels": (
                    st.session_state.cluster_labels.tolist()
                    if st.session_state.cluster_labels is not None
                    else []
                )
            }

            st.download_button(
                "📥 Descargar JSON PCA",
                data=json.dumps(json_data, indent=2),
                file_name="modelo_pca.json"
            )
        else:
            st.info("Aplica PCA primero para exportar sus resultados.")
