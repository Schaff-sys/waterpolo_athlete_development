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

# ---- PAGE CONFIGURATION ----

st.set_page_config(layout="wide")
st.title("Performance Dashboard")
tab1, tab2, tab3 = st.tabs(["Dashboard", "Clustering", "Strain Prediction"])

    # ---- FIX DATA ----
def fix_spanish_decimals(df):
    protected_cols = ["Nombre", "Fecha"]
    for col in df.columns:
        if col not in protected_cols:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(",", ".")
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def file_upload_clean():
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if not uploaded_file:
        st.info("Please upload an Excel file to proceed.")
        st.stop()

    df = pd.read_excel(uploaded_file)

    df = fix_spanish_decimals(df)

    df.columns = df.columns.str.strip()

    st.success("File uploaded successfully")
    return df

df = file_upload_clean()





with tab1:
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


    centre = 1.05
    k = 50
    def ACWR_Penalty(acwr):
        if 0.8 <= acwr <= 1.3:
            return 0
        return k * abs(acwr - centre)

    df['ACWR Penalty'] = df["ACWR"].apply(ACWR_Penalty)
    df['READINESS'] = 50 + 15 * df['HRV_Delta'] - 15 * df['RHR_Delta'] + 10 * df['SLEEP_Delta'] - 10 * df['SORENESS_Delta'] - df['ACWR Penalty']
    df['Readiness Zone'] = pd.cut(df['READINESS'], bins =[0, 30, 50, 70, 90, float('inf')], labels=['Very Poor - Rest', 'Poor - Recovery', 'Fair - Technique work', 'Good - Train', 'Excellent - Overload'])
    st.title (" Acute Chronic Workload Ratio (ACWR) Analysis")

    ## Athlete Selector (Large Display)

    selected_athlete = st.selectbox("Choose Athlete", df['Nombre'].unique(), 
                                    format_func=lambda x: f"{x}")

    ## Latest Scores (Big Cards)

    df_filtered = df[df['Nombre'] == selected_athlete]

    latest = df_filtered.tail(1)


    if not latest.empty:
        acwr = latest['ACWR'].iloc[0]
        zone = latest['Zone'].iloc[0]

        colA, colC = st.columns(2)
        with colA:
            st.metric(
                label="Current ACWR (Predicted injury risk):", 
                value=f"{acwr:.2f} ({zone})", 
                delta=None
            )
            st.caption(f"📅 {latest['Fecha'].dt.strftime('%d/%m').iloc[0]}")

        with colC:
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
                title=f"{selected_athlete} - 28 Day Trend")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if st.button("Compute Clusters"):

        # ----ATHLETE COMPARISON USING CLUSTERING ----
        st.title("Athlete Comparison - Clustering Analysis")

        # Features for clustering (per athlete averages or latest rolling means)
        cluster_features = df.groupby('Nombre').agg({
            'HRV_Rolling': 'last',      # Latest 21-day HRV baseline
            'RHR_Rolling': 'last',      # Latest 21-day RHR baseline  
            'SLEEP_Rolling': 'last',    # Latest 7-day sleep avg
            'SORENESS_Rolling': 'last', # Latest 7-day soreness avg
            'ACWR': 'last',             # Latest workload ratio
            'Training Load': 'mean'     # Avg weekly load tolerance
        }).reset_index()

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

with tab3:
    st.title("Strain prediction")


    # Select only numeric columns (drops non-numeric like 'Nombre', 'Fecha')
    df_numeric = df.select_dtypes(include='number')

    # Now safe correlation
    print(df_numeric.corr()['Strain'].sort_values(ascending=False))

    
    # Select features and target (e.g., predict Strain)
    X = df[['Sleep', 'Training Load']].dropna()  # Inputs
    y = df['Strain'].dropna()  # Output

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

    st.title("Athlete Strain Predictor")
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
    


# ---- CALCULATE WEEKLY METRICS ----

# ---- ADD ATHLETE SELECTION OPTION ----
if "Nombre" not in df.columns:
    st.error("Column 'Nombre' not found in Excel file.")
    st.stop()

athlete_list = (
    df["Nombre"]
    .dropna()
    .astype(str)
    .unique()
    + ['All']
)

athlete = st.selectbox("Select Athlete", sorted(athlete_list))

if athlete == 'All':
    filtered_df = df
else:
    filtered_df = df[df["Nombre"] == athlete]

df["Week"] = df["Fecha"].dt.isocalendar().week

week_num = st.selectbox("Select Week Number", sorted(df["Week"].unique()))

df = df[df["Week"] == week_num]

weekly_summary = df.groupby(["Nombre", "Week"]).agg({
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


st.dataframe(weekly_summary)

