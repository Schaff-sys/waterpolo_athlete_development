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
warnings.filterwarnings("ignore")  # Ignore warnings to keep the app clean

# ---- FIX DATA ----
def fix_spanish_decimals(df):
    """
    Converts columns with comma decimals (common in Spanish Excel files)
    to float values with dot decimals for computation.
    """
    protected_cols = ["Nombre", "Fecha"]  # Columns that shouldn't be converted
    for col in df.columns:
        if col not in protected_cols:
            if df[col].dtype == object:  # Only convert object/string columns
                # Replace commas with dots and convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    return df

def load_and_clean_data(uploaded_file = None):
    """
    Loads Excel file, cleans numeric columns, parses dates, 
    and ensures all required columns exist.
    """
    try:
        if not uploaded_file:
            st.info("Please upload an Excel file to proceed.")
            st.stop()  # Stop execution until file is uploaded

        df = pd.read_excel(uploaded_file, dtype=str)  # Load all as string first

        # Convert Spanish-style decimal columns to floats
        df = fix_spanish_decimals(df)

        # Parse dates and create a month-year column
        df['month_year'] = pd.to_datetime(df['Fecha'], format='ISO8601').dt.strftime('%Y-%m')

        df = df.dropna(subset='month_year')  # Drop rows with missing dates

        # Remove extra spaces in column names
        df.columns = df.columns.str.strip()

        # List of all columns that must exist in the Excel file
        required_columns = ["Nombre", "Fecha", "Lanzamiento 1 (km/h)", "Lanzamiento 2 (km/h)",	
                            "Velocidad de desplazamiento 1 (km/h)", "Velocidad de desplazamiento 2 (km/h)",	
                            "Eficiencia de movimiento (desp)", "Posicion del cuerpo (desp)",	
                            "Conduccion de balon (Seconds)", "Calidad de ejecucción (Cond)",	
                            "Eficiencia del movimiento (Cond)", "Posición del Cuerpo (Cond)",	
                            "Bici fuerza maxima frontal (kg)", "Bici fuerza maxima dorsal (kg)",
                            "50m libre (seconds)", "400m libre (minutes)",	
                            "400m libre (seconds)", "50m braza (seconds)",	
                            "200m braza (mins)", "200m braza (seconds)",	
                            "Press banca (kg)", "Sentadillas (kg)",	
                            "Dominadas", "Velocidad piernas",	
                            "Velocidad brazos", "Velocidad saltos"]

        # Ensure all required columns exist in the uploaded file
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}, Please change the excel file columns")
                st.stop()

        st.success("File uploaded successfully")
        return df
    
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        st.stop()


# Initialize session state variable for uploaded file
if 'file2_df' not in st.session_state:
    st.session_state.file2_df = None

st.title("Athlete Test Management Dashboard")

# Define tabs for different functionalities
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Input", 
    "Individual Metrics", 
    "Rate of Improvement", 
    "Ideal Swim Workloads and Targets", 
    "Ideal Gym Workloads"
])

# ----- TAB 0: File Upload -----
with tab0:
    uploaded_file2 = st.file_uploader("Upload Excel", type=['xlsx'])
    if uploaded_file2 is not None:
        # Load and clean file, store in session_state for other tabs
        st.session_state.file2_df = load_and_clean_data(uploaded_file2)

# ----- TAB 1: Individual Metrics -----
with tab1:
    if st.session_state.file2_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()

    df = st.session_state.file2_df 
    st.subheader("Individual Metrics")

    # Prepare list of columns for tests (exclude identifying columns)
    column_names = df.columns
    remove_columns = ['Nombre', 'Fecha', 'month_year']
    test_list = [col for col in column_names if col not in remove_columns]

    # User selects date and athletes for comparison
    fecha = st.selectbox('Select date of test', sorted(df['month_year'].unique()))
    athlete_list = df["Nombre"].dropna().astype(str).unique().tolist()
    athlete_selected = st.multiselect("Select Athlete", athlete_list)
    test_selected = st.multiselect("Choose test", test_list)

    # Only proceed if athletes and tests are selected
    if athlete_selected and test_selected:
        df_filtered = (df[df["Nombre"].isin(athlete_selected)]
                      .query("month_year == @fecha"))
        
        # Compute average metrics for selected date
        df_avg = df[df['month_year'] == fecha][test_selected].mean().to_frame().T
        df_avg['month_year'] = fecha
        df_avg['Nombre'] = 'Average'  # Label average row

        # Combine individual and average data for plotting
        df_plot = pd.concat([df_filtered[['Nombre'] + test_selected], df_avg])
        
        # Create grouped bar chart
        fig = px.bar(df_plot, 
                    x='Nombre',           # Athletes on x-axis
                    y=test_selected,      
                    barmode='group',
                    title=f"{', '.join(athlete_selected)} - {', '.join(test_selected)} - {fecha}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select athletes and tests")

# ----- TAB 2: Rate of Improvement -----
with tab2: 
    if st.session_state.file2_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()

    df = st.session_state.file2_df 
    st.subheader("Metrics Over Time")

    # List of test columns
    column_names = df.columns
    remove_columns = ['Nombre', 'Fecha', 'month_year']
    test_list = [col for col in column_names if col not in remove_columns]

    athlete_list = df["Nombre"].dropna().astype(str).unique().tolist()
    athlete_list.append('Average')  # Include average as selectable "athlete"
    athlete_selected = st.multiselect("Select Athlete", athlete_list)
    test_selected = st.selectbox("Choose test", test_list)

    # Filter rows with non-null test values
    df_filtered = df[df[test_selected].notna()].copy()  

    # Filter selected athletes
    if athlete_selected:
        real_athletes = [a for a in athlete_selected if a != 'Average']
        if real_athletes:
            df_filtered = df_filtered[df_filtered['Nombre'].isin(real_athletes)]

    # Compute average rows per date if "Average" is selected
    if 'Average' in athlete_selected:
        individual_dates = df_filtered['month_year'].unique()
        for date in individual_dates:
            df_timefiltered = df[df['month_year'] == date]  
            average = df_timefiltered[test_selected].mean(skipna=True)
            if pd.notna(average) and average > 0:  
                avg_row = pd.DataFrame([{
                    'Nombre': 'Average', 
                    test_selected: average, 
                    'month_year': date,
                    'Fecha': df_timefiltered['Fecha'].iloc[0] 
                }], index=[0])
                # Add standard deviation
                stdev_val = df_timefiltered[test_selected].std()
                avg_row[test_selected + '_stdev'] = stdev_val
                df_filtered = pd.concat([df_filtered, avg_row], ignore_index=True)

    # Plot line chart of metrics over time
    fig = px.line(df_filtered, 
                x='Fecha', 
                y=test_selected, 
                color='Nombre',
                title=f"Rate of Improvement: ({', '.join(athlete_selected)} - {test_selected})")
    st.plotly_chart(fig, use_container_width=True)

# ----- TAB 3: Ideal Swim Workloads -----
with tab3:
    if st.session_state.file2_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()

    df = st.session_state.file2_df 
    st.subheader("Ideal Swim Times")

    # Select relevant swim tests
    df_swimtests = df[['Nombre', 'month_year', '50m libre (seconds)', '400m libre (minutes)']]

    # Get latest test per athlete
    latest_tests = df_swimtests.loc[df_swimtests.groupby('Nombre')['month_year'].idxmax()]

    # Convert 400m times in minutes,seconds to seconds
    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return np.nan
        parts = str(time_str).split(',')
        mins = float(parts[0])
        secs = float(parts[1]) if len(parts) > 1 else 0
        return mins * 60 + secs
    
    latest_tests['400m libre (seconds)'] = latest_tests['400m libre (minutes)'].apply(time_to_seconds)
    latest_tests['CSS'] = latest_tests['400m libre (seconds)']/4  # Critical Swim Speed

    # Create quartile groups based on CSS
    top_25_cutoff = latest_tests['CSS'].quantile(0.25)
    mid_top_25_cutoff = latest_tests['CSS'].quantile(0.50)
    mid_bottom_25_cutoff = latest_tests['CSS'].quantile(0.75)
    bottom_25_cutoff = latest_tests['CSS'].quantile(1)

    latest_tests['group'] = pd.cut(latest_tests['CSS'], 
                                bins = [0, top_25_cutoff, mid_top_25_cutoff, 
                                           mid_bottom_25_cutoff, bottom_25_cutoff],
                                labels=['Top 25%', 'Mid-Top 25%', 'Mid-Bottom 25%', 'Bottom 25%'])

    # Aggregate metrics per group
    summary_df = latest_tests.groupby('group').agg({
        'CSS': 'mean',
        '50m libre (seconds)': 'mean',
        'Nombre': lambda x: ', '.join(x.astype(str))
    }).reset_index()

    summary_df.columns = ['Group', 'Avg_CSS', 'Avg_50m', 'Players']

    # Calculate target times for different zones
    summary_df['Zone_1_100_Time'] = summary_df['Avg_CSS'] * 1.15
    summary_df['Zone_2_100_Time'] = summary_df['Avg_CSS'] * 1.05
    summary_df['Zone_3_100_Time'] = summary_df['Avg_CSS']
    summary_df['Zone_4_100_Time'] = summary_df['Avg_CSS'] * 0.9
    summary_df['Zone_5_100_Time'] = summary_df['Avg_50m'] * 2

    # User inputs
    swim_distance = st.selectbox("Select swim distance", ['50m', '100m', '150m', '200m', '300m', '400m'])
    zone_to_work = st.selectbox('Select aerobic zone', ['Zone 1 - Recovery', 'Zone 2 - Aerobic Base', 'Zone 3 - Aerobic Power', 'Zone 4 - Threshold', 'Zone 5 - Neuromuscular'])
   
    swim_distance_value = int(swim_distance[:-1])/100  # Convert to decimal fraction of 100m

    zone_multipliers = {
         'Zone 1 - Recovery': 1.15,
         'Zone 2 - Aerobic Base': 1.05,
         'Zone 3 - Aerobic Power': 1,
         'Zone 4 - Threshold': 0.9,
    }

    # Function to compute target time per athlete group
    def calculate_target_time(row):
        if not zone_to_work == 'Zone 5 - Neuromuscular':
            zone_value = zone_multipliers[zone_to_work]
            target_time = swim_distance_value * zone_value * row['Avg_CSS']
            return target_time
        else:
            target_time = swim_distance_value * row['Avg_50m'] * 2
            return target_time

    # Convert seconds to "minutes:seconds" string
    def convert_time_secs_minutes(seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    # Display metrics for each quartile group
    for idx, row in summary_df.iterrows():
        col1, col2 = st.columns([3, 1])  # 2:1 width ratio
        with col1:
            st.markdown(f"**{row['Group']}**")  # Bold group label
            st.markdown(f"{row['Players']}") 
        with col2:
            st.metric(
                label="Target Time", 
                value=convert_time_secs_minutes(calculate_target_time(row)),
                delta=None
            )
        st.divider()  # Horizontal line

# ----- TAB 4: Ideal Gym Workloads -----
with tab4:
    if st.session_state.file2_df is None:
        st.info("Please upload data in the 'Input' tab to proceed.")
        st.stop()

    df = st.session_state.file2_df 
    st.subheader("Ideal Gym Workloads")

    # Athlete selection
    athlete_list = df["Nombre"].dropna().astype(str).unique().tolist()
    athlete_selected = st.selectbox("Select Athlete", athlete_list)

    # Exercise type selection
    exercise_type = st.selectbox('Select type of exericse:', ["Press banca (kg)", "Sentadillas (kg)"])

    df_gymtests = df[['Nombre', 'month_year', exercise_type]]
    df_gymtests = df_gymtests.dropna(subset=[exercise_type])

    # Get latest test per athlete
    latest_tests = df_gymtests.loc[df_gymtests.groupby('Nombre')['month_year'].idxmax()]
    df_filtered = latest_tests[latest_tests['Nombre'] == athlete_selected].reset_index()

    # User selects type of training goal
    exercise_selected = st.selectbox('Select type of exericse:', ['Endurance', 'Strength', 'Hypertrophy'])

    one_rep_max = df_filtered.loc[0, exercise_type]  # 1RM value

    # Function to calculate recommended weight ranges based on goal
    def get_exercise_weight(df, one_rep_max):
        if exercise_selected == 'Endurance':
            weight_range_max = 0.4 * one_rep_max
            weight_range_min = 0.6 * one_rep_max
            return f"Exercise Weight Range: {weight_range_min}-{weight_range_max}"
         
        elif exercise_selected == 'Strength':
            weight_range_max = 0.3 * one_rep_max
            weight_range_min = 0.7 * one_rep_max
            return f"Exercise Weight Range: {weight_range_min}-{weight_range_max}"

        elif exercise_selected == 'Hypertrophy':
            weight_range_max = 0.7 * one_rep_max
            weight_range_min = 0.9 * one_rep_max
            return f"Exercise Weight Range: {weight_range_min}-{weight_range_max}"

        return None

    weight_info = get_exercise_weight(df_filtered, one_rep_max)

    col1, col2, col3 = st.columns([1, 3, 1])

    # Display exercise details
    with col1:
        st.markdown(f"**{exercise_selected}**")
        st.markdown(f"1RM: **{one_rep_max}kg**")

    with col2:
        # Progress bar and metric for weight range
        min_w, max_w = map(float, weight_info.split(': ')[1].split('-'))
        st.progress((min_w / one_rep_max), text=f"{min_w:.0f}kg")
        st.metric(
            label="🎯 Ideal Weight Range", 
            value=f"{min_w:.0f} - {max_w:.0f}kg"
        )

    with col3:
        # Color-coded rep range guidance
        if exercise_selected == 'Endurance':
            st.markdown("🟢 **15-20 reps**")
        elif exercise_selected == 'Strength':
            st.markdown("🔴 **3-6 reps**")
        else:  # Hypertrophy
            st.markdown("🟡 **8-12 reps**")
