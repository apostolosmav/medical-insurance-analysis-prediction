import pickle as pcl
import pandas as pd
import os
import streamlit as st

try :
    with open('data/raw/Training_insurance.dat','rb') as file:
        best_model =  pcl.load(file)
except Exception as e:
     st.error(f"Datei konnte nicht geladen werden: {e}")
     best_model = None

def predict_cost(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex == 'male' else 0],
        'smoker_yes': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0]
    })
    return best_model.predict(input_df)[0]

# Streamlit UI
st.title("Vorhersage der Versicherungskosten")

age = st.number_input("Alter", min_value=18, max_value=100, value=30)
sex = st.selectbox("Geschlecht", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Kinder", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Raucher", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Kosten berechnen"):
    if best_model:
        prediction = predict_cost(age, sex, bmi, children, smoker, region)
        st.success(f"Geschätzte Kosten: {prediction:.2f} $")
    else:
        st.error("Kein Modell geladen.")

