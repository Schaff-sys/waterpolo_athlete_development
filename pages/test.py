import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Test Result Dashboard")

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

# ---- LOAD DATA ----
df = pd.read_excel(uploaded_file)

df = fix_spanish_decimals(df)

st.success("File uploaded successfully")

# ---- SHOW RAW DATA ----
st.subheader("Raw data")
st.dataframe(df)

# ---- CHECK ATHLETE COLUMN ----
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

# ---- CHECK DATE COLUMN ----
df["Fecha"] = pd.to_datetime(df["Fecha"], format='%d/%m/%Y', errors='coerce')

df['Month_Year'] = df["Fecha"].dt.strftime('%b-%Y').str.lower()

# ---- NUMERIC COLUMNS ----
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found for plotting.")
    st.stop()


y_col = st.selectbox("Y axis", numeric_cols)

fig = px.line(
        df[df["Nombre"] == athlete],
        x="Month_Year",
        y=y_col,
        title=f"{athlete}: {y_col} vs Fecha"
    )

st.plotly_chart(fig, use_container_width=True)