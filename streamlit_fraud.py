import pandas as pd
import numpy as np
import pickle
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# List of possible email domains and billing states based on training data
possible_email_domains = ['example.net', 'example.org']
possible_states = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'IL', 'IN', 'KS', 'KY',
                   'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MT', 'NC', 'ND', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OK', 'OR',
                   'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WY']

# Load and preprocess data
@st.cache
def load_and_preprocess_data():
    # Read CSV and parse the timestamp column
    df = pd.read_csv('Fraud_Amazon.csv')  # Use the correct path to the CSV file

    # Handle Missing Data
    df = df.dropna(subset=["EVENT_LABEL"])  # Drop rows with missing labels

    # Label Encode Target Variable
    label_encoder = LabelEncoder()
    df["EVENT_LABEL"] = label_encoder.fit_transform(df["EVENT_LABEL"])

    # Feature Engineering
    df["EVENT_TIMESTAMP"] = pd.to_datetime(df["EVENT_TIMESTAMP"])
    df["hour"] = df["EVENT_TIMESTAMP"].dt.hour
    df["day_of_week"] = df["EVENT_TIMESTAMP"].dt.dayofweek  # Monday = 0, Sunday = 6
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    df["email_domain"] = df["email_address"].apply(lambda x: x.split('@')[-1])  # Extract domain
    df["email_length"] = df["email_address"].apply(len)  # Length of email address

    df["is_mobile"] = df["user_agent"].apply(lambda x: 1 if "Mobile" in str(x) else 0)
    df["is_old_browser"] = df["user_agent"].apply(lambda x: 1 if re.search(r"Windows NT 5|Windows 98", str(x)) else 0)

    df["phone_valid"] = df["phone_number"].apply(lambda x: 1 if re.match(r"^\(\d{3}\)\d{3}-\d{4}$", str(x)) else 0)

    df["ip_first_octet"] = df["ip_address"].apply(lambda x: int(x.split('.')[0]) if '.' in x else np.nan)

    df["address_length"] = df["billing_address"].apply(len)  # Length of address

    df = df.drop(["EVENT_TIMESTAMP", "user_agent", "email_address", "ip_address", "billing_address", "phone_number", "phone_valid"], axis=1)

    df = pd.get_dummies(df, columns=["email_domain", "billing_state"], drop_first=True)

    return df

# Load preprocessed data
df = load_and_preprocess_data()

# Split data into training and test sets
X = df.drop("EVENT_LABEL", axis=1)
y = df["EVENT_LABEL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('FraudModel.pkl', 'wb') as file:
    pickle.dump(model, file)

# Streamlit UI
st.title('Fraud Detection Model')

# Input fields for prediction
hour = st.number_input('Hour', min_value=0, max_value=23, step=1)
day_of_week = st.number_input('Day of Week (0=Monday, 6=Sunday)', min_value=0, max_value=6, step=1)
is_weekend = st.number_input('Is Weekend (1=Yes, 0=No)', min_value=0, max_value=1)
email_domain = st.text_input('Email Domain')
email_length = st.number_input('Email Length', min_value=1, max_value=100)
is_mobile = st.number_input('Is Mobile (1=Yes, 0=No)', min_value=0, max_value=1)
is_old_browser = st.number_input('Is Old Browser (1=Yes, 0=No)', min_value=0, max_value=1)
ip_first_octet = st.number_input('IP First Octet', min_value=0, max_value=255)
address_length = st.number_input('Billing Address Length', min_value=1, max_value=200)

# Prepare the input data
input_data = {
    'hour': hour,
    'day_of_week': day_of_week,
    'is_weekend': is_weekend,
    'email_length': email_length,
    'is_mobile': is_mobile,
    'is_old_browser': is_old_browser,
    'ip_first_octet': ip_first_octet,
    'address_length': address_length
}

# One-hot encoding for email domain (ensure all possible domains are represented)
for domain in possible_email_domains:
    input_data[f'email_domain_{domain}'] = 1 if email_domain == domain else 0

# One-hot encoding for billing state (ensure all possible states are represented)
for state in possible_states:
    input_data[f'billing_state_{state}'] = 1 if state in email_domain else 0  # Adjust the logic if needed

# Ensure the input has all the columns that were used during training
# Load the model and ensure it expects the same columns
with open('FraudModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Ensure the input has all the required columns (i.e., match the model's feature set)
input_df = pd.DataFrame([input_data])

# Get the model's expected feature columns
model_columns = model.feature_importances_

# Add missing columns (with value 0) to match the model's training data
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match the model's expected order
input_df = input_df[model_columns]

# Make prediction when the button is clicked
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.write("The transaction is predicted to be fraudulent.")
    else:
        st.write("The transaction is predicted to be legitimate.")
    
    st.write("Prediction Probability:", prediction_prob)
