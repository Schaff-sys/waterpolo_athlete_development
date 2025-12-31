import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# ---- PAGE CONFIGURATION ----

st.set_page_config(layout="wide")
st.title("Performance Dashboard")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

# ---- FIX DATA ----

def fix_spanish_decimals(df):
    protected_cols = ["Nombre", "Fecha"]
    for col in df.columns:
        if col not in protected_cols:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(",", ".")
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


if not uploaded_file:
    st.info("Please upload an Excel file to proceed.")
    st.stop()

df = pd.read_excel(uploaded_file)

df = fix_spanish_decimals(df)

df.columns = df.columns.str.strip()

st.success("File uploaded successfully")

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
df['Readiness Zone'] = pd.cut(df['READINESS'], bins =[0, 30, 50, 70, 90, float('inf')], labels=['Very Poor - Rest Day', 'Poor - Light Recovery', 'Fair - Technique/Skill work', 'Good - Train as normal', 'Excellent - Optional Overload'])
st.title (" Acute Chronic Workload Ratio (ACWR) Analysis")

## Athlete Selector (Large Display)

selected_athlete = st.selectbox("Choose Athlete", df['Nombre'].unique(), 
                                   format_func=lambda x: f"{x}")

## Latest Scores (Big Cards)

df = df[df['Nombre'] == selected_athlete]

st.dataframe(df)
latest = df[df['Nombre'] == selected_athlete].tail(1)


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
fig = px.line(df[df['Nombre'] == selected_athlete], 
              x='Fecha', y='ACWR',
              title=f"{selected_athlete} - 28 Day Trend")
st.plotly_chart(fig, use_container_width=True)


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

