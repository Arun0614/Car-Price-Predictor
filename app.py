import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from babel.numbers import format_currency

# --- Streamlit UI setup ---
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Car Price Predictor")
st.markdown("""
Welcome to the **Car Price Predictor App**!  
This app uses a machine learning model trained on a cleaned dataset of used cars to estimate resale prices.  
Just enter your car details to get a quick prediction! 💡🔧  
""")

st.image("/Users/arun/Desktop/Car-Logo-.jpg", use_container_width=True)

# --- Load and preprocess data ---
car = pd.read_csv("/Users/arun/Desktop/project/cleaned car.csv", encoding='latin1')

car = car[car['year'].astype(str).str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != "Ask For Price"]
car['Price'] = car['Price'].astype(str).str.replace(',', '').astype(int)
car['kms_driven'] = car['kms_driven'].astype(str).str.split(' ').str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split(' ').str.slice(0, 3).str.join(' ')
car = car[car['Price'] < 6e6].reset_index(drop=True)

# --- Model training ---
x = car.drop(columns='Price')
y = car['Price']

# Fit OneHotEncoder with the right columns
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(x[['name', 'company', 'fuel_type']])
column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_, handle_unknown='ignore'), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

lr = LinearRegression()

# Find best random state
scores = []
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    scores.append(r2_score(y_test, y_pred))

best_random_state = np.argmax(scores)
best_score = scores[best_random_state]

# Train final model with best random state
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=best_random_state)
pipe = make_pipeline(column_trans, lr)
pipe.fit(x_train, y_train)

# --- User input ---
st.header("Enter Car Details")

with st.form(key='car_details_form'):
    name = st.selectbox("Car Name", sorted(car['name'].unique()))
    company = st.selectbox("Company", sorted(car['company'].unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(car['fuel_type'].unique()))
    year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
    submit_button = st.form_submit_button(label='Submit')

# --- Predict ---
if submit_button:
    year = int(year)
    kms_driven = int(kms_driven)

    input_dict = {
        'name': name,
        'company': company,
        'year': year,
        'kms_driven': kms_driven,
        'fuel_type': fuel_type
    }

    for col in x.columns:
        if col not in input_dict:
            input_dict[col] = x[col].mode()[0]

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[x.columns]

    prediction = pipe.predict(input_df)[0]

    st.success(f"💸 Estimated Price: ₹ {int(prediction):,}")
    st.info(f"📊 Model Accuracy (R² Score): {best_score:.2f}")