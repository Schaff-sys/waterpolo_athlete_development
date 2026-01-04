import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ---- CACHE AND SESSION STATE ----
@st.cache_data
    # ---- FIX DATA ----
def fix_spanish_decimals(df):
            protected_cols = ["Nombre", "Fecha"]
            for col in df.columns:
                if col not in protected_cols:
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(",", ".")
                        df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
def load_and_clean_data(uploaded_file = None):
    try:
        if uploaded_file is None:
            st.info("Please upload an Excel file to proceed.")
            st.stop()
        
        df = pd.read_excel(uploaded_file)

        # Clean data
        df = fix_spanish_decimals(df)

        df["Week"] = df["Fecha"].dt.isocalendar().week

        df.columns = df.columns.str.strip()

        # Check columns exist
        required_columns = ["Nombre", "Fecha", "Duration (min)", "Intensity", "Sleep", "Soreness", "Strain", "Training Load", "HRV", "RHR"]
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                st.stop()

        st.success("File uploaded successfully")
        return df
    
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        st.stop()
    
# ---- SESSION STATE FOR DATAFRAME ----

if 'file1_df' not in st.session_state:
    st.session_state.file1_df = None
# ---- PAGE CONFIGURATION ----

st.set_page_config(layout="wide", page_title="Athlete Strain Management Dashboard")
st.title("Athlete Strain Management Dashboard")
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Input", "ACWR & Readiness", "Summary Metrics", "Strain Groups", "Strain Prediction"])

# ---- KEY CALCULATIONS ----
@st.cache_data
def calculate_acwr_and_readiness(df):
    # ---- CALCULATE ACWR ----
    df.sort_values(['Nombre', 'Fecha']).fillna({'Training Load': 0}, inplace=True)

    df['acute'] = df.groupby('Nombre')['Training Load'].rolling(7, min_periods=1).mean().values
    df['chronic'] = df.groupby('Nombre')['Training Load'].rolling(28, min_periods=1).mean().values 
    df['ACWR'] = df['acute'] / df['chronic']
    df['Zone'] = pd.cut(df['ACWR'], bins =[0, 0.8, 1.3, 1.5, float('inf')], labels=['Low Risk - Increase Load', 'Optimal - Maintain', 'High Risk - Reduce Load', 'Very High Risk - Rest Day'])

    # ---- CALCULATE READINESS SCORE ----
    df['HRV_Rolling'] = df.groupby('Nombre')['HRV'].rolling(21, min_periods=4).mean().values
    df['RHR_Rolling'] = df.groupby('Nombre')['RHR'].rolling(21, min_periods=4).mean().values
    df['SORENESS_Rolling'] = df.groupby('Nombre')['Soreness'].rolling(7, min_periods=3).mean().values
    df['SLEEP_Rolling'] = df.groupby('Nombre')['Sleep'].rolling(7, min_periods=3).mean().values
    df['HRV_Delta'] = (df['HRV'] - df['HRV_Rolling'])/df['HRV_Rolling']
    df['RHR_Delta'] = (df['RHR'] - df['RHR_Rolling'])/df['RHR_Rolling']
    df['SORENESS_Delta'] = (df['Soreness'] - df['SORENESS_Rolling'])/df['SORENESS_Rolling']
    df['SLEEP_Delta'] = (df['Sleep'] - df['SLEEP_Rolling'])/df['SLEEP_Rolling']

    # ---- DEFINE PENALTY FOR ACWR ----
    centre = 1.05
    k = 50
    def ACWR_Penalty(acwr):
        if 0.8 <= acwr <= 1.3:
            return 0
        return k * abs(acwr - centre)
    df['ACWR Penalty'] = df["ACWR"].apply(ACWR_Penalty)
    df['READINESS'] = 50 + 15 * df['HRV_Delta'] - 15 * df['RHR_Delta'] + 10 * df['SLEEP_Delta'] - 10 * df['SORENESS_Delta'] - df['ACWR Penalty']
    df['Readiness Zone'] = pd.cut(df['READINESS'], bins =[0, 30, 50, 70, 90, float('inf')], labels=['Very Poor - Rest', 'Poor - Recovery', 'Fair - Technique work', 'Good - Train', 'Excellent - Overload'])
    return df

with tab0:
    uploaded_file1 = st.file_uploader("Upload Excel file 1", type=["xlsx"])

    if uploaded_file1 is not None:
        st.session_state.file1_df = load_and_clean_data(uploaded_file1)

with tab1:
    if st.session_state.file1_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()
    df = calculate_acwr_and_readiness(st.session_state.file1_df)
    st.session_state.file1_df = df    
    # ---- DASHBOARD DISPLAY ----
    st.subheader ("ACWR and Readiness Dashboard")

    ## Athlete Selector for readiness dashboard (Large Display)

    selected_athlete = st.selectbox("Choose Athlete", df['Nombre'].unique(), 
                                    format_func=lambda x: f"{x}")
    df_filtered = df[df['Nombre'] == selected_athlete]
    latest = df_filtered.tail(1)


    if not latest.empty:
        acwr = latest['ACWR'].iloc[0]
        zone = latest['Zone'].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Current ACWR (Predicted injury risk):", 
                value=f"{acwr:.2f} ({zone})", 
                delta=None
            )
            st.caption(f"📅 {latest['Fecha'].dt.strftime('%d/%m').iloc[0]}")

        with col2:
            Readiness_value = latest['READINESS'].iloc[0]
            Readiness_zone = latest['Readiness Zone'].iloc[0]
            st.metric(
                label="Readiness Score:", 
                value=f"{Readiness_value:.2f} ({Readiness_zone})", 
                delta=None
            )
    
    ## Trend Chart Below
    fig = px.line(df_filtered, 
                x='Fecha', y='ACWR',
                title=f"{selected_athlete} - 28 Day ACWR Trend")
    ## Adding horizontal lines for ACWR zones
    fig.add_hrect(y0=0, y1=0.8, fillcolor="green", opacity=0.1, line_width=0, layer="below",
                  annotation_text="Low Risk - Increase Load", annotation_position="top left")
    
    fig.add_hrect(y0=0.8, y1=1.3, fillcolor="blue", opacity=0.1, line_width=0, layer="below",
                  annotation_text="Optimal - Maintain", annotation_position="top left")
    
    fig.add_hrect(y0=1.3, y1=1.5, fillcolor="orange", opacity=0.1, line_width=0, layer="below",
                  annotation_text="High Risk - Reduce Load", annotation_position="top left")
    
    fig.add_hrect(y0=1.5, y1=df_filtered['ACWR'].max()+0.5, fillcolor="red", opacity=0.1, line_width=0, layer="below",
                  annotation_text="Very High Risk - Rest Day", annotation_position="top left")
    
        # Estilizar anotaciones
    fig.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        font=dict(size=12),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if st.session_state.file1_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()
    df = calculate_acwr_and_readiness(st.session_state.file1_df)
    
    st.subheader ("Summary Metrics Dashboard")
    # ---- ADD ATHLETE SELECTION OPTION ----
    if "Nombre" not in df.columns:
        st.error("Column 'Nombre' not found in Excel file.")
        st.stop()

    athlete_list = (
        df["Nombre"]
        .dropna()
        .astype(str)
        .unique()
    )
  
    athlete_list = athlete_list.tolist()
   

    athlete_selected = st.multiselect("Select Athlete", athlete_list)

    if athlete_selected:
        df_filtered = df[df["Nombre"].isin(athlete_selected)]
    else:
        df_filtered = df.head(0)

    # ---- FEATURE SELECTION ----
    feature_list = []
    feature_list.extend(['Duration (min)', 'Intensity', 'Sleep', 'Soreness', 'Strain', 'Training Load'])
    feature_select = st.multiselect("Feature selection", feature_list)
    
    # ---- WEEK RANGE SELECTION ----
    available_weeks = sorted(df["Week"].dropna().unique())
    week_min, week_max = st.slider("Select Week Number", min_value=min(available_weeks), max_value=max(available_weeks), value=(min(available_weeks), max(available_weeks)))
    df_filtered = df_filtered[(df_filtered["Week"] >= week_min) & (df_filtered["Week"] <= week_max)]
    weekly_summary = df_filtered.groupby(["Nombre", "Week"]).agg({
        'Training Load': ['sum','std', 'mean'],
        'Strain': 'sum',
        'Sleep': 'mean',
        'Soreness': 'mean'
    }).round(2).reset_index()

    weekly_summary.columns = ['Nombre', 'Week', 'Total_Load', 'Stdev_Load', 'Avg_Load', 'Total_Strain', 'Avg_Sleep', 'Avg_Soreness']

    weekly_summary['Avg_Daily_Load'] = weekly_summary['Total_Load'] / 7
    weekly_summary["Monotony"] = weekly_summary['Avg_Daily_Load'] / weekly_summary['Stdev_Load']
    weekly_summary['Sleep_Adjusted'] = 1 + ((weekly_summary['Avg_Sleep'] - 1/(10 - 1)) * 4)
    weekly_summary['Weekly_Strain'] = weekly_summary['Total_Strain'] * weekly_summary['Monotony']
    weekly_summary['Avg_Wellness'] = (weekly_summary['Sleep_Adjusted'] + (6-weekly_summary['Avg_Soreness'])) / 2
    weekly_summary['Adjusted_Strain'] = weekly_summary['Weekly_Strain'] * ((6-weekly_summary['Avg_Wellness'])/5)

    weekly_summary['Adjusted_to_Actual_Difference'] = weekly_summary['Weekly_Strain'] - weekly_summary['Adjusted_Strain']

    if not df_filtered.empty:
    ## Trend Chart Below
        fig = px.line(df_filtered, 
                    x='Fecha', y=feature_select, color='Nombre',
                    title=f"{', '.join(athlete_selected)} - from Week {week_min} to {week_max} Trend")
        fig.update_layout(
            yaxis_title=f"{', '.join(feature_select)}")

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please select at least one athlete to display the trend chart and weekly summary.")



with tab3:
    if st.session_state.file1_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()
    df = calculate_acwr_and_readiness(st.session_state.file1_df)
    
    if st.button("Compute Clusters"):

        # Features for clustering (per athlete averages or latest rolling means)
        cluster_features = df.groupby('Nombre').agg({
            'HRV_Rolling': 'last',      # Latest 21-day HRV baseline
            'RHR_Rolling': 'last',      # Latest 21-day RHR baseline  
            'SLEEP_Rolling': 'last',    # Latest 7-day sleep avg
            'SORENESS_Rolling': 'last', # Latest 7-day soreness avg
            'ACWR': 'last',             # Latest workload ratio
            'Training Load': 'mean'     # Avg weekly load tolerance
        }).reset_index()

        if len(cluster_features) < 3:
            st.error("Not enough athletes with complete data for clustering. Please ensure at least 3 athletes have sufficient data.")
            st.stop()

        # Standardize for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_features.drop('Nombre', axis=1))

        # Cluster into 3-4 groups
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_features['Cluster'] = kmeans.fit_predict(features_scaled)

        st.subheader("🏅 Athlete Training Recommendations by Cluster")

        # Define cluster advice
        cluster_advice = {
            0: {
                "name": "High Recovery 🟢",
                "advice": """
                **Load High Recovery athletes harder:**
                - ACWR up to 1.5 safe
                - 5-6 training days/week  
                - Higher intensity sessions
                - Less recovery needed
                """
            },
            1: {
                "name": "Load Sensitive 🔴", 
                "advice": """
                **Protect Load Sensitive athletes:**
                - ACWR < 1.2 maximum
                - 3-4 training days/week
                - More recovery/rest days
                - Technique focus over volume
                """
            },
            2: {
                "name": "Consistent Workers 🟡",
                "advice": """
                **Standard programming:**
                - ACWR 0.8-1.3 target
                - 4-5 training days/week
                - Balanced intensity/volume
                - Monitor weekly trends
                """
            }
        }

    
        # Dropdowns for each cluster
        list_of_clusters = sorted(cluster_features['Cluster'].unique().tolist())
        def athletes_in_cluster(cluster_id):
            return cluster_features[cluster_features['Cluster'] == cluster_id]['Nombre']
        
        
        def find_cluster_name(cluster):
                return cluster_advice[cluster]['name']
        
        for cluster in list_of_clusters:
            st.subheader(f"Cluster: {find_cluster_name(cluster)}")

            athletes_list = athletes_in_cluster(cluster).tolist()  # List of strings
            athletes_str = ", ".join(athletes_list)
            st.write(f"Athletes: {athletes_str}")

            st.write(cluster_advice[cluster]['advice'])

with tab4:
    if st.session_state.file1_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()
    df = calculate_acwr_and_readiness(st.session_state.file1_df)


    # Select only numeric columns (drops non-numeric like 'Nombre', 'Fecha')
    df_numeric = df.select_dtypes(include='number')

    # Now safe correlation
    print(df_numeric.corr()['Strain'].sort_values(ascending=False))

    
    # Select features and target (e.g., predict Strain)
    X = df[['Sleep', 'Training Load']].dropna()  # Inputs
    y = df['Strain'].dropna()  # Output

    if len(X) < 10 or len(y) < 10:
        st.error("Not enough data to train the prediction model. Please ensure sufficient data is available.")
        st.stop()

    # Align X and y to have same number of rows
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Results
    print("Intercept:", model.intercept_)
    print("Coefficients (Sleep, Training Load):", model.coef_)
    print("R² on train:", model.score(X_train, y_train))
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Test R²:", r2_score(y_test, y_pred_test))
    print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

    # ---- USER INPUT FOR PREDICTION ----

    st.subheader("Athlete Strain Predictor")
    sleep = st.slider("Sleep (hours)", 6.0, 10.0, 8.0)
    duration = st.slider("Duration (minutes)", 30, 240, 60)
    intensity = st.slider("Intensity (1-5)", 1, 5, 1)
    tl = duration * intensity
    pred = model.predict([[sleep, tl]])[0]

            
      # Color-coded alert
    col1, col2 = st.columns([3,1])
    with col1:
        st.metric("Predicted Strain", f"{pred:.3f}")
    with col2:
        if pred > 0.5: st.error("⚠️ HIGH - Reduce Load")
        elif pred > 0.3: st.warning("📈 MODERATE - Maintain Load") 
        else: st.success("✅ LOW - Increase Load")
    


