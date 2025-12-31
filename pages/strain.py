import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")
st.title("Performance Dashboard")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

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

# ---- LOAD AND FIX DATA ----
df = pd.read_excel(uploaded_file)

df = fix_spanish_decimals(df)

df.columns = df.columns.str.strip()

st.success("File uploaded successfully")

# ---- SHOW RAW DATA ----
st.subheader("Raw data")
st.dataframe(df)

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

athlete = st.selectbox("Select Athlete", sorted(athlete_list))

# ---- ADD WEEK SELECTION OPTION ----

df = df[df["Nombre"] == athlete]

df["Week"] = df["Fecha"].dt.isocalendar().week

week_num = st.selectbox("Select Week Number", sorted(df["Week"].unique()))

df = df[df["Week"] == week_num]


# ---- CALCULATE WEEKLY METRICS ----

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

# ---- CALCULATE ACWR ----
df.sort_values(['Nombre', 'Fecha']).fillna({'Training Load': 0}, inplace=True)

df['acute'] = df.groupby('Nombre')['Training Load'].rolling(7, min_periods=1).sum().values
df['chronic'] = df.groupby('Nombre')['Training Load'].rolling(28, min_periods=1).mean().values 
df['ACWR'] = df['acute'] / df['chronic']
df['Zone'] = pd.cut(df['ACWR'], bins =[0, 0.8, 1.3, 1.5, float('inf')], labels=['Low Risk', 'Optimal', 'High Risk', 'Very High Risk'])

st.title ("Acute:Chronic Workload Ratio (ACWR) Analysis")

## Athlete Selector (Large Display)
col1, col2 = st.columns([3, 1])
with col1:
    selected_athlete = st.selectbox("Choose Athlete", df['Nombre'].unique(), 
                                   format_func=lambda x: f"🏅 {x}")
with col2:
    st.metric("Refresh", "Live", delta="↻")

## Latest Scores (Big Cards)
latest = df[df['Nombre'] == selected_athlete].tail(1)
if not latest.empty:
    acwr = latest['ACWR'].iloc[0]
    zone = latest['Zone'].iloc[0]
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(
            label="ACWR", 
            value=f"{acwr:.2f}", 
            delta=None
        )
        st.caption(f"📅 {latest['Fecha'].dt.strftime('%d/%m').iloc[0]}")

    with colB:
        st.markdown(f"### {zone}")
    with colC:
        st.metric("Training Load", latest['Training Load'].iloc[0], 
                 delta=latest['acute'].iloc[0] - latest['chronic'].iloc[0])

## Trend Chart Below
fig = px.line(df[df['Nombre'] == selected_athlete], 
              x='Fecha', y=['ACWR', 'Training Load'],
              title=f"{selected_athlete} - 28 Day Trend")
st.plotly_chart(fig, use_container_width=True)

## Team Overview Table
st.subheader("All Athletes - Latest Status")
latest_all = df.groupby('Nombre').tail(1)[['Nombre', 'Fecha', 'ACWR', 'Zone', 'Training Load']]
st.dataframe(latest_all.sort_values('ACWR', ascending=False), 
             column_config={"Zone": st.column_config.ColorColumn("Risk Zone")},
             use_container_width=True)