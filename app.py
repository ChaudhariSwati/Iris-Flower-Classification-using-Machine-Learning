import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("src/iris_model.pkl", "rb"))

st.title("Iris Flower Prediction")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):
    prediction = model.predict([[sl, sw, pl, pw]])
    st.success(f"The predicted species is: {prediction[0]}")
