# insurance_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pcl

# --- Load Dataset ---
df = pd.read_csv('insurance.csv')

# --- Load Model ---
with open('Training_insurance.dat','rb') as file:
    best_model = pcl.load(file)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["EDA & Insights", "Prediction & Interaction"])

# --- Section 1: EDA & Insights ---
if section == "EDA & Insights":
    st.title("Exploratory Data Analysis & Insights")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())

    # Missing values
    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    # Distribution of Charges
    st.subheader("Distribution of Charges")
    fig, ax = plt.subplots()
    sns.histplot(df['charges'], bins=20, kde=True, ax=ax, color='skyblue')
    st.pyplot(fig)

    # Charges by Smoker
    st.subheader("Charges by Smoker Status")
    fig, ax = plt.subplots()
    sns.boxplot(x='smoker', y='charges', data=df, ax=ax, palette=['lightgreen','lightcoral'])
    st.pyplot(fig)

    # Charges by Sex
    st.subheader("Charges by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x='sex', y='charges', data=df, ax=ax, palette=['lightblue','lightpink'])
    st.pyplot(fig)

    # Charges by Region
    st.subheader("Charges by Region")
    fig, ax = plt.subplots()
    sns.boxplot(x='region', y='charges', data=df, ax=ax, palette='pastel')
    st.pyplot(fig)

    # Scatter plots: BMI vs Charges colored by smoker
    st.subheader("BMI vs Charges")
    fig, ax = plt.subplots()
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, palette={'yes':'red','no':'green'}, alpha=0.6)
    st.pyplot(fig)

    # Scatter plots: Age vs Charges colored by smoker
    st.subheader("Age vs Charges")
    fig, ax = plt.subplots()
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df, palette={'yes':'red','no':'green'}, alpha=0.6)
    st.pyplot(fig)

    # Interaction: Smoker x BMI
    st.subheader("Average Charges by Smoker and BMI Category")
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,100], labels=['Underweight','Normal','Overweight','Obese'])
    bmi_smoker_avg = df.groupby(['smoker','bmi_category'])['charges'].mean().unstack()
    st.dataframe(bmi_smoker_avg)
    bmi_smoker_avg.plot(kind='bar', figsize=(10,5))
    st.pyplot(plt.gcf())

    # Interaction: Smoker x Age
    st.subheader("Average Charges by Smoker and Age Group")
    df['age_group'] = pd.cut(df['age'], bins=[17,29,39,49,59,69,79,89,100],
                             labels=['18-29','30-39','40-49','50-59','60-69','70-79','80-89','90-100'])
    age_smoker_avg = df.groupby(['smoker','age_group'])['charges'].mean().unstack()
    st.dataframe(age_smoker_avg)
    age_smoker_avg.plot(kind='bar', figsize=(12,5))
    st.pyplot(plt.gcf())

    # Summary Insights
    st.subheader("Key Insights")
    st.markdown("""
    - **Smokers** have significantly higher insurance charges than non-smokers.
    - **BMI** is strongly correlated with charges; obese smokers are the highest payers.
    - **Age** increases insurance costs gradually; costs rise sharply after ~50, especially for smokers.
    - **Sex** has minor effect on charges.
    - **Region** has minimal impact.
    - Interactions between **smoker status and BMI** or **smoker status and age** are strong drivers of charges.
    """)

# --- Prediction Function ---
def predict_cost(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame({
        'age':[age],
        'bmi':[bmi],
        'children':[children],
        'sex_male':[1 if sex=='male' else 0],
        'smoker_yes':[1 if smoker=='yes' else 0],
        'region_northwest':[1 if region=='northwest' else 0],
        'region_southeast':[1 if region=='southeast' else 0],
        'region_southwest':[1 if region=='southwest' else 0]
    })
    return best_model.predict(input_df)[0]

# --- Section 2: Prediction & Interaction ---
if section == "Prediction & Interaction":
    st.title("Insurance Charges Prediction & Trends")

    # Common inputs
    children = st.slider("Children", 0, 10, 0)
    sex = st.selectbox("Sex", ["male", "female"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    # --- Dynamic Charges vs BMI Curve ---
    st.subheader("Predicted Charges vs BMI")
    age_for_bmi = st.slider("Select Age for BMI Trend", 18, 100, 30)
    bmi_range = list(range(10, 61))
    charges_smoker = [predict_cost(age_for_bmi, sex, b, children, "yes", region) for b in bmi_range]
    charges_nonsmoker = [predict_cost(age_for_bmi, sex, b, children, "no", region) for b in bmi_range]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(bmi_range, charges_smoker, color='red', label='Smoker')
    ax.plot(bmi_range, charges_nonsmoker, color='green', label='Non-Smoker')
    ax.set_xlabel("BMI")
    ax.set_ylabel("Predicted Charges")
    ax.set_title(f"Predicted Charges vs BMI at Age {age_for_bmi}")
    ax.legend()
    st.pyplot(fig)

    # --- Dynamic Charges vs Age Curve ---
    st.subheader("Predicted Charges vs Age")
    bmi_for_age = st.slider("Select BMI for Age Trend", 10.0, 60.0, 25.0)
    age_range = list(range(18, 101))
    charges_smoker_age = [predict_cost(a, sex, bmi_for_age, children, "yes", region) for a in age_range]
    charges_nonsmoker_age = [predict_cost(a, sex, bmi_for_age, children, "no", region) for a in age_range]

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(age_range, charges_smoker_age, color='red', label='Smoker')
    ax2.plot(age_range, charges_nonsmoker_age, color='green', label='Non-Smoker')
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Predicted Charges")
    ax2.set_title(f"Predicted Charges vs Age at BMI {bmi_for_age}")
    ax2.legend()
    st.pyplot(fig2)

    # --- Specific Prediction ---
    st.subheader("Predict Specific Charges")
    age_input = st.slider("Age for Specific Prediction", 18, 100, 30)
    bmi_input = st.slider("BMI for Specific Prediction", 10.0, 60.0, 25.0)
    smoker_input = st.selectbox("Smoker for Specific Prediction", ["yes", "no"])

    if st.button("Predict Specific Charges"):
        pred = predict_cost(age_input, sex, bmi_input, children, smoker_input, region)
        st.success(f"Predicted Charges: ${pred:.2f}")

        # Comparison smoker vs non-smoker
        smoker_pred = predict_cost(age_input, sex, bmi_input, children, "yes", region)
        nonsmoker_pred = predict_cost(age_input, sex, bmi_input, children, "no", region)
        st.info(f"Smoker: ${smoker_pred:.2f}, Non-Smoker: ${nonsmoker_pred:.2f}")
        st.warning(f"Difference (Smoker - Non-Smoker): ${smoker_pred - nonsmoker_pred:.2f}")


