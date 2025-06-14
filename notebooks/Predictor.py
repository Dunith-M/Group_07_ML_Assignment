import numpy as np
import pandas as pd
import streamlit as st
#streamlit run Predictor.py
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load('model.pkl')

# Load data
df = pd.read_csv('../datasets/Student Depression Dataset.csv')
df.columns = df.columns.str.strip()  # remove accidental whitespace in column names

# Title
st.title('🧠 Student Depression Predictor')

# Label Encoders
le_gender = LabelEncoder().fit(df['Gender'])
le_city = LabelEncoder().fit(df['City'])
le_prof = LabelEncoder().fit(df['Profession'])
le_sleep = LabelEncoder().fit(df['Sleep Duration'])
le_dietary = LabelEncoder().fit(df['Dietary Habits'])
le_degree = LabelEncoder().fit(df['Degree'])
le_suicide = LabelEncoder().fit(df['Have you ever had suicidal thoughts ?'])
le_family = LabelEncoder().fit(df['Family History of Mental Illness'])

# Input fields for categorical values
gender = st.selectbox('Gender', le_gender.classes_)
city = st.selectbox('City', le_city.classes_)
profession = st.selectbox('Profession', le_prof.classes_)
sleep_duration = st.selectbox('Sleep Duration', le_sleep.classes_)
dietary = st.selectbox('Dietary Habits', le_dietary.classes_)
degree = st.selectbox('Degree', le_degree.classes_)
suicidal_thoughts = st.selectbox('Have you ever had suicidal thoughts?', le_suicide.classes_)
family_history = st.selectbox('Family History of Mental Illness', le_family.classes_)

# Input fields for numerical values
age = st.number_input('Age', min_value=18, max_value=100)
academic_pressure = st.slider('Academic Pressure', 0, 5)
cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, step=0.1)
study_satisfaction = st.slider('Study Satisfaction', 0, 5)
job_satisfaction = st.slider('Job Satisfaction', 0, 5)
work_study_hours = st.slider('Work/Study Hours', 0, 5)
financial_pressure = st.slider('Financial Pressure', 0, 5)

# Predict button
if st.button('Predict'):
    input_data = np.array([
        le_gender.transform([gender])[0],
        le_city.transform([city])[0],
        le_prof.transform([profession])[0],
        age,
        academic_pressure,
        cgpa,
        study_satisfaction,
        job_satisfaction,
        le_sleep.transform([sleep_duration])[0],
        le_dietary.transform([dietary])[0],
        le_degree.transform([degree])[0],
        le_suicide.transform([suicidal_thoughts])[0],
        work_study_hours,
        financial_pressure,
        le_family.transform([family_history])[0]
    ]).reshape(1, -1)

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success('✅ The student is **not** depressed 😊')
    else:
        st.error('⚠️ The student **is likely depressed** 😢')
