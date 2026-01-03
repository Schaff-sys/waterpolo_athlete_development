import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ---- CACHE & SESSION STATE ----
@st.cache_data
def load_and_clean_data(uploaded_file):
    """Carga y limpia datos de Excel con manejo robusto de errores."""
    try:
        df = pd.read_excel(uploaded_file)
        if df.empty:
            st.error("❌ Archivo vacío.")
            st.stop()
            
        df.columns = df.columns.str.strip()
        
        # Validar columnas críticas
        required_cols = ['Nombre', 'Fecha']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Columnas faltantes: {missing_cols}")
            st.stop()
            
        # Convertir Fecha
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        
        # Fix decimales españoles
        df = fix_spanish_decimals(df)
        df['Week'] = df['Fecha'].dt.isocalendar().week
        
        st.success("✅ Archivo cargado correctamente")
        return df
        
    except Exception as e:
        st.error(f"❌ Error cargando archivo: {str(e)}")
        st.stop()

def fix_spanish_decimals(df):
    """Corrige decimales con coma española."""
    protected_cols = ["Nombre", "Fecha"]
    for col in df.columns:
        if col not in protected_cols and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float, errors='ignore')
    return df

# ---- CONFIGURACIÓN ----
st.set_page_config(layout="wide", page_title="Performance Dashboard")
st.title("🏋️ Performance Dashboard")

# ---- SESSION STATE ----
if 'df' not in st.session_state:
    st.session_state.df = None

# ---- TABS ----
tab0, tab1, tab2, tab3 = st.tabs(["📁 Input", "📊 Dashboard", "🔍 Clustering", "🔮 Strain Prediction"])

# ---- TAB 0: CARGA DE DATOS ----
with tab0:
    uploaded_file = st.file_uploader("Sube archivo Excel", type=["xlsx"])
    if uploaded_file is not None and st.session_state.df is None:
        with st.spinner("Procesando datos..."):
            st.session_state.df = load_and_clean_data(uploaded_file)
    
    if st.session_state.df is not None:
        st.info(f"✅ Datos cargados: {len(st.session_state.df)} filas, {len(st.session_state.df['Nombre'].unique())} atletas")

# ---- FUNCIÓN PRINCIPAL: CÁLCULOS ----
@st.cache_data
def calculate_metrics(df):
    """Calcula todas las métricas con manejo de errores."""
    df = df.copy()
    df.sort_values(['Nombre', 'Fecha'], inplace=True)
    
    # Rellenar NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # ACWR
    df['acute'] = df.groupby('Nombre')['Training Load'].transform(
        lambda x: x.rolling(7, min_periods=1).mean())
    df['chronic'] = df.groupby('Nombre')['Training Load'].transform(
        lambda x: x.rolling(28, min_periods=1).mean())
    df['ACWR'] = np.where(df['chronic'] > 0, df['acute'] / df['chronic'], 1.0)
    
    df['Zone'] = pd.cut(df['ACWR'], 
                       bins=[0, 0.8, 1.3, 1.5, float('inf')], 
                       labels=['🟢 Low Risk', '🟡 Optimal', '🟠 High Risk', '🔴 Very High Risk'])
    
    # Readiness rolling metrics
    rollings = {
        'HRV_Rolling': ('HRV', 21, 4),
        'RHR_Rolling': ('RHR', 21, 4),
        'SORENESS_Rolling': ('Soreness', 7, 3),
        'SLEEP_Rolling': ('Sleep', 7, 3)
    }
    
    for new_col, (base_col, window, min_periods) in rollings.items():
        if base_col in df.columns:
            df[new_col] = df.groupby('Nombre')[base_col].transform(
                lambda x: x.rolling(window, min_periods=min_periods).mean())
    
    # Deltas
    for base, rolling in [('HRV', 'HRV_Rolling'), ('RHR', 'RHR_Rolling'), 
                         ('Soreness', 'SORENESS_Rolling'), ('Sleep', 'SLEEP_Rolling')]:
        rolling_col = f"{rolling}"
        if rolling_col in df.columns:
            df[f'{base}_Delta'] = np.where(
                df[rolling_col] > 0, 
                (df[base] - df[rolling_col]) / df[rolling_col], 0)
    
    # ACWR Penalty
    def acwr_penalty(acwr):
        centre, k = 1.05, 50
        if 0.8 <= acwr <= 1.3:
            return 0
        return k * abs(acwr - centre)
    
    df['ACWR Penalty'] = df['ACWR'].apply(acwr_penalty)
    df['READINESS'] = (50 + 
                      (15 * df.get('HRV_Delta', 0)) - 
                      (15 * df.get('RHR_Delta', 0)) + 
                      (10 * df.get('SLEEP_Delta', 0)) - 
                      (10 * df.get('Soreness_Delta', 0)) - 
                      df['ACWR Penalty'])
    
    df['Readiness Zone'] = pd.cut(df['READINESS'], 
                                 bins=[0, 30, 50, 70, 90, float('inf')],
                                 labels=['🔴 Very Poor', '🟠 Poor', '🟡 Fair', '🟢 Good', '💚 Excellent'])
    
    return df

# ---- TAB 1: DASHBOARD ----
with tab1:
    if st.session_state.df is None:
        st.warning("⚠️ Por favor carga datos en la pestaña 'Input'")
        st.stop()
    
    df = calculate_metrics(st.session_state.df)
    st.session_state.df = df  # Actualizar cache
    
    st.subheader("📈 Análisis ACWR & Readiness")
    
    # Selector atleta
    selected_athlete = st.selectbox("👤 Selecciona atleta", df['Nombre'].unique())
    df_filtered = df[df['Nombre'] == selected_athlete]
    latest = df_filtered.tail(1)
    
    if not latest.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ACWR", f"{latest['ACWR'].iloc[0]:.2f}", 
                     delta=None, label=f"📅 {latest['Fecha'].dt.strftime('%d/%m').iloc[0]}")
            st.caption(latest['Zone'].iloc[0])
        
        with col2:
            st.metric("Readiness", f"{latest['READINESS'].iloc[0]:.1f}", 
                     delta=None)
            st.caption(latest['Readiness Zone'].iloc[0])
    
    # Gráfico tendencia
    fig = px.line(df_filtered, x='Fecha', y='ACWR', 
                  title=f"{selected_athlete} - Tendencia ACWR 28 días")
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekly summary
    st.subheader("📋 Resumen Semanal")
    athlete_list = df["Nombre"].drop_duplicates().tolist()
    athlete_selected = st.multiselect("Atletas", athlete_list, default=athlete_list[:3])
    
    features = ['Training Load', 'Sleep', 'Soreness', 'Strain']
    feature_select = st.multiselect("Métricas", features, default=['Training Load'])
    
    weeks = sorted(df["Week"].dropna().unique())
    week_range = st.slider("Semanas", min(weeks), max(weeks), (min(weeks), max(weeks)))
    
    df_week = df[df['Nombre'].isin(athlete_selected)]
    df_week = df_week[(df_week["Week"] >= week_range[0]) & 
                     (df_week["Week"] <= week_range[1])]
    
    if not df_week.empty and feature_select:
        fig = px.line(df_week, x='Fecha', y=feature_select, 
                     color='Nombre', title=f"{' | '.join(athlete_selected)}")
        st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: CLUSTERING ----
with tab2:
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
        st.stop()
    
    if st.button("🔍 Calcular Clusters", type="primary"):
        with st.spinner("Calculando clusters..."):
            df = calculate_metrics(st.session_state.df)
            
            # Features para clustering
            cluster_features = df.groupby('Nombre').agg({
                'HRV_Rolling': 'last', 'RHR_Rolling': 'last',
                'SLEEP_Rolling': 'last', 'SORENESS_Rolling': 'last',
                'ACWR': 'last', 'Training Load': 'mean'
            }).reset_index()
            
            if len(cluster_features) < 2:
                st.error("❌ Necesitas al menos 2 atletas")
                st.stop()
            
            # Clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(cluster_features.drop('Nombre', axis=1))
            kmeans = KMeans(n_clusters=min(3, len(cluster_features)), random_state=42)
            cluster_features['Cluster'] = kmeans.fit_predict(features_scaled)
            
            st.subheader("🏅 Recomendaciones por Cluster")
            
            cluster_advice = {
                0: ("🟢 Alta Recuperación", "Aumenta carga hasta ACWR 1.5, 5-6 días/semana"),
                1: ("🔴 Sensible a Carga", "ACWR < 1.2 máximo, 3-4 días/semana"),
                2: ("🟡 Consistente", "ACWR 0.8-1.3, 4-5 días/semana")
            }
            
            for cluster_id in sorted(cluster_features['Cluster'].unique()):
                athletes = cluster_features[cluster_features['Cluster'] == cluster_id]['Nombre'].tolist()
                name, advice = cluster_advice.get(cluster_id, ("Desconocido", "Monitorear"))
                
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.subheader(f"{name}")
                    st.write(f"**Atletas:** {', '.join(athletes)}")
                with col2:
                    st.info(advice)

# ---- TAB 3: PREDICCIÓN ----
with tab3:
    if st.session_state.df is None:
        st.warning("⚠️ Carga datos primero")
        st.stop()
    
    df = calculate_metrics(st.session_state.df)
    
    # Entrenar modelo
    @st.cache_data
    def train_strain_model(_df):
        X = _df[['Sleep', 'Training Load']].dropna()
        y = _df['Strain'].dropna()
        
        if len(X) < 10:
            return None, "Datos insuficientes"
        
        min_len = min(len(X), len(y))
        X, y = X.iloc[:min_len], y.iloc[:min_len]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        
        r2 = r2_score(y_test, model.predict(X_test))
        return model, f"R² = {r2:.3f}"
    
    model, model_info = train_strain_model(df)
    
    if model is None:
        st.error(model_info)
        st.stop()
    
    st.metric("📊 Calidad del Modelo", model_info)
    
    # Predicción interactiva
    col1, col2, col3 = st.columns(3)
    with col1:
        sleep = st.slider("💤 Horas sueño", 6.0, 10.0, 8.0)
    with col2:
        duration = st.slider("⏱️ Duración (min)", 30, 240, 60)
    with col3:
        intensity = st.slider("🔥 Intensidad (1-5)", 1, 5, 3)
    
    training_load = duration * intensity / 100  # Normalizar
    pred_strain = model.predict([[sleep, training_load]])[0]
    
    colA, colB = st.columns(2)
    with colA:
        st.metric("🎯 Strain Predicho", f"{pred_strain:.3f}")
    
    with colB:
        if pred_strain > 0.5:
            st.error("⚠️ ALTO - Reduce carga")
        elif pred_strain > 0.3:
            st.warning("📈 MODERADO - Mantén carga")
        else:
            st.success("✅ BAJO - Aumenta carga")

