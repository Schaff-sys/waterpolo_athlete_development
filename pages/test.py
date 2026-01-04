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


    # ---- FIX DATA ----
def fix_spanish_decimals(df):
            protected_cols = ["Nombre", "Fecha"]
            for col in df.columns:
                if col not in protected_cols:
                    if df[col].dtype == object:
                        df[col] = pd.to_numeric(df[col].astype(str()).str.replace(',', '.'), errors='coerce')
                        #df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
def load_and_clean_data(uploaded_file = None):
    try:
        if not uploaded_file:
            st.info("Please upload an Excel file to proceed.")
            st.stop()
        df = pd.read_excel(uploaded_file, dtype=str)

        # Clean data
        df = fix_spanish_decimals(df)

        df.to_csv('test')

        # Parse dates then format as "MM-YYYY"
        df['month_year'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y').dt.strftime('%Y-%m')

        df = df.dropna(subset='month_year')

        df.columns = df.columns.str.strip()

        # Check columns exist
        required_columns = ["Nombre", "Fecha", "Lanzamiento 1 (km/h)",	"Lanzamiento 2 (km/h)",	
                            "Velocidad de desplazamiento 1 (km/h)",	"Velocidad de desplazamiento 2 (km/h)",	
                            "Eficiencia de movimiento (desp)",	"Posicion del cuerpo (desp)",	
                            "Conduccion de balon (Seconds)",	"Calidad de ejecucción (Cond)",	
                            "Eficiencia del movimiento (Cond)", "Posición del Cuerpo (Cond)",	
                            "Bici fuerza maxima frontal (kg)",	"Bici fuerza maxima dorsal (kg)",
                            "50 m libre (seconds)",	"400 m libre (minutes)",	
                            "400m libre (seconds)",	"50m braza (seconds)",	
                            "200m braza (mins)",	"200m braza (seconds)",	
                            "Press banca (kg)",	"Sentadillas (kg)",	
                            "Dominadas", "Velocidad piernas",	
                            "Velocidad brazos",	"Velocidad saltos"]
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                st.stop()

        st.success("File uploaded successfully")
        return df
    
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        st.stop()

# Before tabs - initialize if missing
if 'file2_df' not in st.session_state:
    st.session_state.file2_df = None

# Your tabs code...
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Input", "Individual Metrics", "Time Progression", "Groups", "Scatter Plots", "Ideal Workloads", "Standard Deviation", "Shooting Values"])

with tab0:
    uploaded_file2 = st.file_uploader("Upload Excel", type=['xlsx'])
    if uploaded_file2 is not None:
        st.session_state.file2_df = load_and_clean_data(uploaded_file2)

    st.dataframe(st.session_state.file2_df)

with tab1:
    if st.session_state.file2_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()

    df = st.session_state.file2_df 
    st.subheader("Individual Metrics")

    column_names = df.columns
    remove_columns = ['Nombre', 'Fecha', 'month_year']
    test_list = [col for col in column_names if col not in remove_columns]

    fecha = st.selectbox('Select date of test', sorted(df['month_year'].unique()))
    athlete_list = df["Nombre"].dropna().astype(str).unique().tolist()
    athlete_selected = st.multiselect("Select Athlete", athlete_list)
    test_selected = st.multiselect("Choose test", test_list)

    # Filter AFTER all selections
    if athlete_selected and test_selected:
        df_filtered = (df[df["Nombre"].isin(athlete_selected)]
                      .query("month_year == @fecha"))
        
        # Calculate averages for selected date and tests
        df_avg = df[df['month_year'] == fecha][test_selected].mean().to_frame().T
        df_avg['month_year'] = fecha
        df_avg['Nombre'] = 'Average'  # Label averages distinctly

        df_plot = pd.concat([df_filtered[['Nombre'] + test_selected], df_avg])
        
        fig = px.bar(df_plot, 
                    x='Nombre',           # Athletes on x-axis
                    y=test_selected,      # Multiple test columns
                    barmode='group',
                    title=f"Test results - {fecha}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select athletes and tests")

with tab2: 
        if st.session_state.file2_df is None:
            st.info("Please upload data in the 'Input' tab to proceed.")
            st.stop()

        df = st.session_state.file2_df 
        st.subheader("Metrics Over Time")

        column_names = df.columns
        remove_columns = ['Nombre', 'Fecha', 'month_year']
        test_list = [col for col in column_names if col not in remove_columns]

        athlete_list = df["Nombre"].dropna().astype(str).unique().tolist()
        athlete_list.append('Average')
        athlete_selected = st.multiselect("Select Athlete", athlete_list)
        test_selected = st.selectbox("Choose test", test_list)

    # START FRESH - Single filtering flow
        df_filtered = df[df[test_selected].notna()].copy()  # Test filter first

        # Filter athletes SECOND (on already test-filtered data)
        if athlete_selected:
            real_athletes = [a for a in athlete_selected if a != 'Average']
            if real_athletes:
                df_filtered = df_filtered[df_filtered['Nombre'].isin(real_athletes)]

        # Add Average rows LAST - use filtered dates only
        if 'Average' in athlete_selected:
            individual_dates = df_filtered['month_year'].unique()  # Use FILTERED dates
            for date in individual_dates:
                df_timefiltered = df[df['month_year'] == date]  # Full df for calculation
                average = df_timefiltered[test_selected].mean(skipna=True)
                if pd.notna(average) and average > 0:  # Valid average only
                    avg_row = pd.DataFrame([{
                        'Nombre': 'Average', 
                        test_selected: average, 
                        'month_year': date,
                        'Fecha': df_timefiltered['Fecha'].iloc[0]  # Need for plotting
                    }], index=[0])
                    stdev_val = df_timefiltered[test_selected].std()
                    avg_row[test_selected + '_stdev'] = stdev_val
                    df_filtered = pd.concat([df_filtered, avg_row], ignore_index=True)


        # Plot works with all data
        fig = px.line(df_filtered, 
                    x='Fecha', 
                    y=test_selected, 
                    color='Nombre',
                    title=f"Test results ({test_selected}) - {', '.join(athlete_selected)}")  # Fixed: join() converts list to string
        st.plotly_chart(fig, use_container_width=True)


with tab3:
    if st.session_state.file2_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()
    
    if st.button("Compute Clusters"):
        # Select numeric columns for clustering (exclude names/dates)
        numeric_cols = [
            'Lanzamiento 1 (km/h)', 'Lanzamiento 2 (km/h)', 
            'Velocidad de desplazamiento 1 (km/h)', 'Velocidad de desplazamiento 2 (km/h)',
            'Eficiencia de movimiento (desp)', 'Posicion del cuerpo (desp)',
            'Conduccion de balon (Seconds)', 'Calidad de ejecucción (Cond)',
            'Bici fuerza maxima frontal (kg)', 'Bici fuerza maxima dorsal (kg)',
            '50 m libre (seconds)', 'Press banca (kg)', 'Sentadillas (kg)'
        ]

        st.dataframe(df[['Velocidad de desplazamiento 1 (km/h)']])

        for cols in numeric_cols:
            df[cols] = pd.to_numeric(df[cols], errors='coerce')

        grouped_df = df.groupby('Nombre')[numeric_cols].mean()
        st.dataframe(grouped_df)

        # Use only tests with >70% complete data
        complete_pct = grouped_df[numeric_cols].notna().mean()
        good_cols = complete_pct[complete_pct > 0.7].index.tolist()

        df_cluster = grouped_df[good_cols].copy()
        df_cluster = df_cluster.fillna(df_cluster.median())  # Median preserves sports distributions

        # Clean and prepare data
        df_cluster = grouped_df[numeric_cols].copy()
        df_cluster = df_cluster.dropna()  # Remove rows with any missing values
        df_cluster = df_cluster.astype(float)  # Ensure all numeric

        # Standardize features (critical for k-means)
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_cluster), 
                                columns=numeric_cols, 
                                index=df_cluster.index)

        # Find optimal clusters (elbow method)
        inertias = []
        K_range = range(2, 10)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df_scaled)
            inertias.append(kmeans.inertia_)

        # Plot elbow curve
        fig_elbow = px.line(x=list(K_range), y=inertias, 
                        title="Elbow Method - Optimal Clusters")
        st.plotly_chart(fig_elbow)

