# Insurance Charges Prediction

**Predict medical insurance costs and explore key drivers using Python and Streamlit.**

---

## **Project Overview**

This project analyzes medical insurance charges and builds predictive models to estimate costs based on key factors such as:

- **Age** of the insured person  
- **Body Mass Index (BMI)**  
- **Gender**  
- **Number of children**  
- **Smoker status**  
- **Region**  

The project combines **exploratory data analysis (EDA)**, **statistical modeling**, and an **interactive Streamlit app** to provide actionable insights and predictions.

**Main Features:**

1. **Exploratory Data Analysis (EDA)**  
   - Visualizations of insurance charges by smoker status, gender, region, BMI, and age.  
   - Identification of key patterns and trends in the dataset.  
   - Interactive group analyses (e.g., average charges by smoker and BMI category).  

2. **Predictive Modeling**  
   - Multiple regression models are trained and evaluated:  
     - Linear Regression  
     - Decision Tree Regressor  
     - Random Forest Regressor  
     - Gradient Boosting Regressor  
   - Model evaluation metrics include **R²** and **Mean Absolute Error (MAE)**.  
   - The best-performing model is saved and used in the Streamlit app for predictions.  

3. **Streamlit Interactive App**  
   - Users can input personal information (age, BMI, children, smoker status, gender, region) to predict insurance charges.  
   - Dynamic visualizations show predicted charges versus BMI or Age.  
   - Side-by-side comparison of **smoker vs non-smoker** charges for the same parameters.

4. ## **Presentation**

- [View Presentation (PDF)](presentation/Medical_Insurance_Prediction_Presentation.pdf)  
- [Download Presentation (PPTX)](presentation/Medical_Insurance_Prediction_Presentation.pptx)


---

## **Installation**

### 1. Clone the repository:

```bash
git clone https://github.com/apostolosmav/medical-insurance-analysis-prediction
cd insurance-charges-prediction
