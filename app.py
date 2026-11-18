import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate

# 1. Setup page config (Unchanged)
st.set_page_config(page_title="Actuarial Pricing Engine", layout="wide")
st.title("🩺 Health Insurance Pricing Engine (GLM)")
st.write("""
This app demonstrates a standard actuarial pricing model using a **Gamma GLM with a log-link function**. 
Adjust the policyholder characteristics below to see how the base premium is modified by specific rating factors.
""")

# 2. Load Data via DuckDB SQL and Train Model (Cached for performance)
@st.cache_resource
def train_model():
    # Force smoker to stay 'yes'/'no' text instead of DuckDB auto-converting to boolean
    sql_query = """
    WITH base AS (
        SELECT 
            age,
            CAST(sex AS VARCHAR) AS sex,
            bmi,
            children,
            CASE WHEN CAST(smoker AS VARCHAR) IN ('true', 'True', 'yes', '1') THEN 'yes' ELSE 'no' END AS smoker,
            CAST(region AS VARCHAR) AS region,
            charges,
            CASE WHEN age < 30 THEN '18-29' WHEN age < 45 THEN '30-44' 
                 WHEN age < 60 THEN '45-59' ELSE '60+' END AS age_band
        FROM read_csv_auto('insurance.csv', all_varchar=False)
    )
    SELECT age, sex, bmi, children, smoker, region, charges,
           ROUND(AVG(bmi) OVER (PARTITION BY smoker, age_band), 2) AS cohort_avg_bmi
    FROM base
    """
    df = duckdb.query(sql_query).df()
    
    # Define features (Unchanged)
    categorical_features = ['sex', 'smoker', 'region']
    numeric_features = ['age', 'bmi', 'children']
    
    # Create preprocessing pipeline (Unchanged)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit Gamma GLM (Tweedie with power=2 is Gamma)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', TweedieRegressor(power=2, alpha=0.5, link='log', max_iter=1000))
    ])
    
    X = df[categorical_features + numeric_features]
    y = df['charges']
    model.fit(X, y)
    
    # Extract Base Rate (Intercept) and Rating Factors (Coefficients) (Unchanged)
    regressor = model.named_steps['regressor']
    intercept = regressor.intercept_
    coefs = regressor.coef_
    
    cat_names = model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(categorical_features)
    all_features = list(cat_names) + numeric_features
    
    base_rate = np.exp(intercept)
    rating_factors = dict(zip(all_features, np.exp(coefs)))

    # 5-Fold CV Benchmark & Gradient Boosting fit for the resume claims
    gbm_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(loss='gamma', max_depth=3, random_state=42))
    ])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    glm_cv = cross_validate(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    gbm_cv = cross_validate(gbm_pipe, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    glm_mae = -np.mean(glm_cv['test_score'])
    gbm_mae = -np.mean(gbm_cv['test_score'])
    gbm_pipe.fit(X, y)
    
    return model, base_rate, rating_factors, gbm_pipe, all_features, glm_mae, gbm_mae, sql_query

model, base_rate, rating_factors, gbm_pipe, all_features, glm_mae, gbm_mae, sql_query = train_model()

# 3. User Interface (Sidebar for Inputs - Unchanged)
st.sidebar.header("Policyholder Profile")
age = st.sidebar.slider("Age", 18, 64, 30)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0, 0.1)
children = st.sidebar.slider("Dependents (Children)", 0, 5, 0)
sex = st.sidebar.selectbox("Sex", ["female", "male"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# 4. Process Inputs and Calculate Premium (Unchanged)
input_data = pd.DataFrame({
    'age': [age], 'bmi': [bmi], 'children': [children],
    'sex': [sex], 'smoker': [smoker], 'region': [region]
})

predicted_premium = model.predict(input_data)[0]

# 5. Display Results (Unchanged)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Quoted Annual Premium")
    st.metric(label="Predicted Pure Premium", value=f"${predicted_premium:,.2f}")
    st.caption("Calculated as: Base Rate × (Age Factor ^ Age) × (BMI Factor ^ BMI) ... × Categorical Factors")

with col2:
    st.subheader("Under the Hood: Rating Factors")
    st.write(f"**Base Rate:** ${base_rate:,.2f}")
    
    applied_factors = {}
    applied_factors['Age Modification'] = rating_factors['age'] ** age
    applied_factors['BMI Modification'] = rating_factors['bmi'] ** bmi
    
    if sex == 'male': applied_factors['Sex (Male)'] = rating_factors['sex_male']
    if smoker == 'yes': applied_factors['Smoker (Yes)'] = rating_factors['smoker_yes']
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

# 6. Optional Dropdown at the Bottom (Backs up SQL, 5-Fold CV & SHAP on Resume)
with st.expander("🔍 View SQL Pipeline, 5-Fold CV Benchmark & SHAP Attribution"):
    mae_diff_pct = ((glm_mae - gbm_mae) / glm_mae) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Gamma GLM 5-Fold CV MAE", f"${glm_mae:,.2f}")
    c2.metric("Gradient Boosting 5-Fold CV MAE", f"${gbm_mae:,.2f}")
    c3.metric("Out-of-Fold MAE Reduction", f"{mae_diff_pct:.1f}%")

    X_trans = gbm_pipe.named_steps['preprocessor'].transform(input_data)
    explainer = shap.TreeExplainer(gbm_pipe.named_steps['regressor'])
    shap_values = explainer(pd.DataFrame(X_trans, columns=all_features))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    shap.plots.waterfall(shap_values[0], max_display=6, show=False)
    st.pyplot(fig)

    st.code(sql_query, language="sql")
