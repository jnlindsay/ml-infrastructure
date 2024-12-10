import streamlit as st
import polars as pl
import numpy as np
import mlflow
from mlflow.models import infer_signature
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.write("*Note*: this is a dummy notebook that does not aim to produce meaningful results.")

st.write("Download data: `kaggle competitions download -c playground-series-s4e12`")

df_train = pl.read_csv('kaggle/data/playground-series-s4e12/train.csv').drop_nulls()
df_test = pl.read_csv('kaggle/data/playground-series-s4e12/test.csv').drop_nulls()

st.write("Training data:")
df_train

st.write("Test data:")
df_test

# plotly chart
fig = px.scatter(
  df_train[["Annual Income", "Premium Amount"]],
  x = "Annual Income",
  y = "Premium Amount"
)
st.plotly_chart(fig)