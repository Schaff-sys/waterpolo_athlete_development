import streamlit as st
import plotly.express as px

def render(df):
    st.header("Athlete Strain")
    athlete = st.selectbox("Athlete", df["athlete"].unique())
    fig = px.line(df[df["athlete"] == athlete], x="date", y="RPE")
    st.plotly_chart(fig, use_container_width=True)
