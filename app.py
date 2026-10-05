import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Setup page config
st.set_page_config(page_title="Actuarial Pricing Engine", layout="wide")
st.title("🩺 Health Insurance Pricing Engine (GLM)")
st.write("""
This app demonstrates a standard actuarial pricing model using a **Gamma GLM with a log-link function**. 
Adjust the policyholder characteristics below to see how the base premium is modified by specific rating factors.
""")

# 2. Load Data and Train Model (Cached for performance)
@st.cache_resource
def train_model():
    # Load Kaggle medical cost dataset
    df = pd.read_csv("insurance.csv")
    
    # Define features
    categorical_features = ['sex', 'smoker', 'region']
    numeric_features = ['age', 'bmi', 'children']
    
    # Create preprocessing pipeline (dummy coding for categorical)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit Gamma GLM (Tweedie with power=2 is Gamma)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', TweedieRegressor(power=2, alpha=0.5, link='log'))
    ])
    
    X = df.drop('charges', axis=1)
    y = df['charges']
    model.fit(X, y)
    
    # Extract Base Rate (Intercept) and Rating Factors (Coefficients)
    regressor = model.named_steps['regressor']
    intercept = regressor.intercept_
    coefs = regressor.coef_
    
    # Get feature names from one-hot encoder
    cat_names = model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(categorical_features)
    all_features = list(cat_names) + numeric_features
    
    # Exponentiate intercept for Base Rate, and coefficients for Rating Factors
    base_rate = np.exp(intercept)
    rating_factors = dict(zip(all_features, np.exp(coefs)))
    
    return model, base_rate, rating_factors

model, base_rate, rating_factors = train_model()

# 3. User Interface (Sidebar for Inputs)
st.sidebar.header("Policyholder Profile")
age = st.sidebar.slider("Age", 18, 64, 30)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0, 0.1)
children = st.sidebar.slider("Dependents (Children)", 0, 5, 0)
sex = st.sidebar.selectbox("Sex", ["female", "male"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# 4. Process Inputs and Calculate Premium
input_data = pd.DataFrame({
    'age': [age], 'bmi': [bmi], 'children': [children],
    'sex': [sex], 'smoker': [smoker], 'region': [region]
})

# Predict using the pipeline
predicted_premium = model.predict(input_data)[0]

# 5. Display Results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Quoted Annual Premium")
    st.metric(label="Predicted Pure Premium", value=f"${predicted_premium:,.2f}")
    st.caption("Calculated as: Base Rate × (Age Factor ^ Age) × (BMI Factor ^ BMI) ... × Categorical Factors")

with col2:
    st.subheader("Under the Hood: Rating Factors")
    st.write(f"**Base Rate:** ${base_rate:,.2f}")
    
    # Calculate exact multipliers applied for this specific user
    applied_factors = {}
    
    # Age, BMI, Children are continuous (Factor ^ Value)
    applied_factors['Age Modification'] = rating_factors['age'] ** age
    applied_factors['BMI Modification'] = rating_factors['bmi'] ** bmi
    
    # Categorical lookups
    if sex == 'male': applied_factors['Sex (Male)'] = rating_factors['sex_male']
    if smoker == 'yes': applied_factors['Smoker (Yes)'] = rating_factors['smoker_yes']
    # Dynamically check if the region has a multiplier (ignores the dropped baseline)
    if f'region_{region}' in rating_factors:
        applied_factors[f'Region ({region.capitalize()})'] = rating_factors[f'region_{region}']
        
    factor_df = pd.DataFrame([
        {"Factor": k, "Multiplier": f"{v:.4f}x"} 
        for k, v in applied_factors.items()
    ])
    st.table(factor_df)

st.divider()
st.write("### Model Insights")
st.write(f"Notice the massive rating factor for smoking. In this GLM, being a smoker multiplies the base premium by **{rating_factors['smoker_yes']:.2f}x**, holding all other variables constant.")
