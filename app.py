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
# Configuración inicial
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
        return pd.read_csv(archivo)
    except Exception as e:
        st.error(f"❌ Error leyendo el archivo: {e}")
        return None


# Si no sube archivo, se detiene
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
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

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

X = df.drop(columns=[target_column]).values
y = df[target_column].values
feature_names = df.drop(columns=[target_column]).columns.tolist()
target_names = np.unique(y)

# ---------------------
# SessionState
# ---------------------
ss = st.session_state

for key, value in {
    "modelo_supervisado": None,
    "modelo_no_supervisado": None,
    "metricas_supervisado": {},
    "metricas_no_supervisado": {},
    "X_test": None,
    "y_test": None,
    "y_pred": None,
    "X_pca": None,
    "cluster_labels": None
}.items():
    ss.setdefault(key, value)

# ---------------------
# Menú principal
# ---------------------
modo = st.sidebar.radio(
    "Selecciona el modo:",
    ["🏠 Inicio", "📊 Modo Supervisado", "🔍 Modo No Supervisado", "💾 Exportación"]
)

# =========================================================
# 🏠 MODO INICIO
# =========================================================
if modo == "🏠 Inicio":
    st.header("📊 Bienvenido al Análisis de Salud Mental y Redes Sociales")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Registros", f"{len(df):,}")
    col2.metric("Características", len(feature_names))
    col3.metric("Clases Únicas", len(target_names))
    col4.metric("Objetivo", target_column)

    st.subheader("📌 Vista previa del dataset")
    st.dataframe(df_original.head(10), use_container_width=True)

    st.subheader("📑 Descripción rápida")
    st.write(df_original.describe(include="all").transpose())


# =========================================================
# 📊 MODO SUPERVISADO
# =========================================================
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
        test_size = st.slider("Tamaño test (%)", 10, 40, 20) / 100
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)

    if st.button("🚀 Entrenar Modelo Supervisado", type="primary"):
        with st.spinner("Entrenando modelo..."):
            try:
                # CORREGIDO: random_state correcto
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )

                ss.X_test = X_test
                ss.y_test = y_test

                modelo = DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )

                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

                ss.y_pred = y_pred

                metricas = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                }

                ss.modelo_supervisado = modelo
                ss.metricas_supervisado = metricas

                st.success("✅ Modelo entrenado exitosamente!")
            except Exception as e:
                st.error(f"Error al entrenar modelo: {e}")

    if ss.modelo_supervisado:
        st.markdown("---")
        st.subheader("📈 Métricas del Modelo")

        m = ss.metricas_supervisado
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{m['accuracy']:.4f}")
        col2.metric("Precision", f"{m['precision']:.4f}")
        col3.metric("Recall", f"{m['recall']:.4f}")
        col4.metric("F1-score", f"{m['f1_score']:.4f}")

        # ---------- Importancia de características ----------
        try:
            modelo = ss.modelo_supervisado
            importances = modelo.feature_importances_
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False)

            st.subheader("📊 Importancia de características")
            fig_fi = px.bar(fi_df, x="feature", y="importance", title="Feature Importances")
            st.plotly_chart(fig_fi, use_container_width=True)
        except:
            pass

        # ---------- Matriz de confusión ----------
        try:
            y_test = ss.y_test
            y_pred = ss.y_pred

            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=[str(x) for x in target_names], columns=[str(x) for x in target_names])

            st.subheader("🔢 Matriz de Confusión")
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_df.values,
                x=cm_df.columns,
                y=cm_df.index,
                hoverongaps=False
            ))
            fig_cm.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        except:
            pass

        # ---------- Predicción manual ----------
        st.markdown("---")
        st.subheader("🧪 Predicción Manual")

        manual_input = {}

        for feat in feature_names:
            col_values = df[feat]
            if pd.api.types.is_numeric_dtype(col_values):
                val = st.number_input(feat, float(col_values.min()), float(col_values.max()), float(col_values.mean()))
                manual_input[feat] = val
            else:
                uniques = sorted(list(np.unique(col_values)))
                val = st.selectbox(feat, uniques)
                manual_input[feat] = val

        if st.button("🔮 Predecir"):
            try:
                arr = np.array([manual_input[f] for f in feature_names]).reshape(1, -1)
                pred = ss.modelo_supervisado.predict(arr)[0]

                st.success(f"Predicción: {pred}")
                st.json({"input": manual_input, "prediction": int(pred)})
            except Exception as e:
                st.error(f"Error en la predicción: {e}")


# =========================================================
# 🔍 MODO NO SUPERVISADO
# =========================================================
elif modo == "🔍 Modo No Supervisado":
    st.header("🔍 PCA + Clustering")

    max_components = min(len(feature_names), 20)
    n_components = st.slider("Número de Componentes PCA", 2, max_components, 3)
    n_clusters = st.slider("Clusters (KMeans)", 2, 10, 3)

    if st.button("🚀 Aplicar PCA y Clustering", type="primary"):
        from sklearn.cluster import KMeans

        with st.spinner("Aplicando PCA..."):
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_pca)

                metricas = {
                    "silhouette_score": float(silhouette_score(X_pca, cluster_labels)),
                    "davies_bouldin": float(davies_bouldin_score(X_pca, cluster_labels)),
                    "variance_explained": pca.explained_variance_ratio_.tolist(),
                    "total_variance": float(sum(pca.explained_variance_ratio_))
                }

                ss.modelo_no_supervisado = pca
                ss.metricas_no_supervisado = metricas
                ss.X_pca = X_pca
                ss.cluster_labels = cluster_labels

                st.success("✅ PCA aplicado correctamente")
            except Exception as e:
                st.error(f"Error: {e}")

    if ss.modelo_no_supervisado:
        st.markdown("---")
        st.subheader("📈 Métricas PCA + KMeans")

        m = ss.metricas_no_supervisado
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette", f"{m['silhouette_score']:.4f}")
        col2.metric("Davies-Bouldin", f"{m['davies_bouldin']:.4f}")
        col3.metric("Varianza Total", f"{m['total_variance']:.4f}")

        # Scree plot
        try:
            scree_df = pd.DataFrame({
                "component": [f"PC{i+1}" for i in range(len(m['variance_explained']))],
                "variance": m["variance_explained"]
            })
            st.subheader("📈 Scree Plot (Varianza Explicada)")
            st.plotly_chart(px.bar(scree_df, x="component", y="variance"), use_container_width=True)
        except:
            pass

        # PCA scatter
        try:
            X_pca = ss.X_pca
            labels = ss.cluster_labels

            pca_df = pd.DataFrame({
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1] if X_pca.shape[1] > 1 else np.zeros(len(X_pca)),
                "cluster": labels.astype(str)
            })

            st.subheader("🔎 Scatter Plot (PC1 vs PC2)")
            st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="cluster"), use_container_width=True)
        except:
            pass


# =========================================================
# 💾 EXPORTACIÓN
# =========================================================
elif modo == "💾 Exportación":
    st.header("💾 Exportación de Modelos")

    col1, col2 = st.columns(2)

    # ----------- EXPORTACIÓN SUPERVISADO -----------
    with col1:
        st.subheader("📊 Modelo Supervisado")

        if ss.modelo_supervisado:
            modelo = ss.modelo_supervisado

            json_data = {
                "model_type": "Supervised",
                "model_name": "Decision Tree (Gini)",
                "target_variable": target_column,
                "metrics": ss.metricas_supervisado,
                "features": feature_names,
                "feature_importances": modelo.feature_importances_.tolist()
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
            st.info("Entrena el modelo supervisado primero.")

    # ----------- EXPORTACIÓN PCA -----------
    with col2:
        st.subheader("🔍 Modelo No Supervisado")

        if ss.modelo_no_supervisado:
            json_data = {
                "model_type": "Unsupervised",
                "algorithm": "PCA + KMeans",
                "metrics": ss.metricas_no_supervisado,
                "cluster_labels": ss.cluster_labels.tolist() if ss.cluster_labels is not None else []
            }

            st.download_button(
                "📥 Descargar JSON PCA",
                data=json.dumps(json_data, indent=2),
                file_name="modelo_pca.json"
            )
        else:
            st.info("Aplica PCA primero.")
