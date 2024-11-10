import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


data = pd.read_csv("Verizon Data.csv")



X = data.drop('default', axis=1)
y = data['default']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, 'default_model.pkl')


model = joblib.load('default_model.pkl')


st.markdown("""
    <style>
        .stApp {
            background-color: #1B0130; 
            color: #FFFFFF; 
            background-image: url("/Users/moulang/Desktop/Projects/stars.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .sidebar .sidebar-content {
            background-color: #2D033B;
        }
        .css-1d391kg {
            background-color: #1B0130; 
        }
        .stButton>button {
            background-color: #6A0DAD; 
            color: #FFFFFF;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #8E2EED; 
        }
    </style>
""", unsafe_allow_html=True)




st.title("💜 MAX Consulting: Customer Default Prediction App")
st.markdown("**Delivering insights powered by machine learning and cutting-edge analytics.**")


st.sidebar.header("Enter Customer Details")


year = st.sidebar.number_input("Year", min_value=2020, max_value=2030, value=2020)
month = st.sidebar.selectbox("Month", range(1, 13), index=0)
day = st.sidebar.selectbox("Day", range(1, 32), index=0)
price = st.sidebar.number_input("Price ($)", min_value=0.0, step=0.01)
downpmt = st.sidebar.number_input("Down Payment ($)", min_value=0.0, step=0.01)
monthdue = st.sidebar.slider("Months Due", min_value=0, max_value=24, value=6)
payment_left = st.sidebar.number_input("Remaining Payment ($)", min_value=0.0, step=0.01)
monthly_payment = st.sidebar.number_input("Monthly Payment ($)", min_value=0.0, step=0.01)
pmttype = st.sidebar.selectbox("Payment Type", [1, 2, 3, 4, 5])
credit_score = st.sidebar.slider("Credit Score", min_value=0, max_value=8, value=6)
age = st.sidebar.number_input("Age (Years)", min_value=0, max_value=120, step=1)
gender = st.sidebar.radio("Gender", [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')


input_data = pd.DataFrame([{
    'year': year,
    'month': month,
    'day': day,
    'price': price,
    'downpmt': downpmt,
    'monthdue': monthdue,
    'payment_left': payment_left,
    'monthly_payment': monthly_payment,
    'pmttype': pmttype,
    'credit_score': credit_score,
    'age': age,
    'gender': gender
}])

if st.button("💡 Predict"):
    prediction = model.predict(input_data)
    result = "🔴 Default" if prediction[0] == 1 else "🟢 No Default"

    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"**Prediction: {result}**")
        st.warning("⚠️ The customer is likely to default. Consider reviewing their payment plan.")
    else:
        st.success(f"**Prediction: {result}**")
        st.balloons() 
        st.info("✅ The customer is unlikely to default. Payment history appears stable.")


st.markdown("""
    <hr style="border:1px solid #8E2EED">
    <center>Powered by <strong>MAX Consulting</strong> | Delivering Excellence with Insights</center>
""", unsafe_allow_html=True)
