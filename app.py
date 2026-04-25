import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Rebuild model directly in app
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500
    ages = np.random.randint(27, 39, n)
    frequent_flyer = np.random.choice(['Yes', 'No'], n)
    income_class = np.random.choice(['Low Income', 'Middle Income', 'High Income'], n)
    services = np.random.randint(1, 7, n)
    social_media = np.random.choice(['Yes', 'No'], n)
    hotel = np.random.choice(['Yes', 'No'], n)

    target = []
    for i in range(n):
        score = 0
        if frequent_flyer[i] == 'Yes': score += 1
        if income_class[i] == 'High Income': score += 1
        if services[i] >= 4: score += 1
        if social_media[i] == 'Yes': score += 1
        if hotel[i] == 'Yes': score += 1
        target.append(1 if score >= 3 else 0)

    df = pd.DataFrame({
        'Age': ages,
        'FrequentFlyer': frequent_flyer,
        'AnnualIncomeClass': income_class,
        'ServicesOpted': services,
        'AccountSyncedToSocialMedia': social_media,
        'BookedHotelOrNot': hotel,
        'Target': target
    })

    le = LabelEncoder()
    for col in ['FrequentFlyer', 'AnnualIncomeClass', 
                'AccountSyncedToSocialMedia', 'BookedHotelOrNot']:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Target', axis=1)
    y = df['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# App UI
st.title("✈️ Customer Churn Prediction")
st.markdown("### Predict whether a customer will churn or not")
st.markdown("---")

st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", min_value=18, max_value=70, value=30)
frequent_flyer = st.sidebar.selectbox("Frequent Flyer?", options=['Yes', 'No'])
annual_income = st.sidebar.selectbox("Annual Income Class", 
                    options=['Low Income', 'Middle Income', 'High Income'])
services_opted = st.sidebar.slider("Number of Services Opted", 
                    min_value=1, max_value=6, value=3)
social_media = st.sidebar.selectbox("Account Synced to Social Media?", 
                    options=['Yes', 'No'])
hotel_booked = st.sidebar.selectbox("Booked Hotel or Not?", 
                    options=['Yes', 'No'])

# Encode inputs
frequent_flyer_enc = 1 if frequent_flyer == 'Yes' else 0
income_enc = {'Low Income': 1, 'Middle Income': 2, 'High Income': 0}[annual_income]
social_enc = 1 if social_media == 'Yes' else 0
hotel_enc = 1 if hotel_booked == 'Yes' else 0

input_data = pd.DataFrame({
    'Age': [age],
    'FrequentFlyer': [frequent_flyer_enc],
    'AnnualIncomeClass': [income_enc],
    'ServicesOpted': [services_opted],
    'AccountSyncedToSocialMedia': [social_enc],
    'BookedHotelOrNot': [hotel_enc]
})

st.subheader("Customer Details Summary")
st.write(f"**Age:** {age}")
st.write(f"**Frequent Flyer:** {frequent_flyer}")
st.write(f"**Annual Income Class:** {annual_income}")
st.write(f"**Services Opted:** {services_opted}")
st.write(f"**Account Synced to Social Media:** {social_media}")
st.write(f"**Booked Hotel:** {hotel_booked}")
st.markdown("---")

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN!")
        st.write(f"**Churn Probability: {round(probability[1]*100, 2)}%**")
    else:
        st.success(f"✅ This customer is likely to STAY!")
        st.write(f"**Retention Probability: {round(probability[0]*100, 2)}%**")

    st.markdown("---")
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Outcome': ['Will Stay', 'Will Churn'],
        'Probability': [round(probability[0]*100, 2), 
                       round(probability[1]*100, 2)]
    })
    st.dataframe(prob_df)

st.markdown("---")
st.caption("Customer Churn Prediction | Random Forest Model | BTech CS Project")
