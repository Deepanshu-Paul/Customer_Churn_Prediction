import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model

#load the trained model, Scalar pickle file, label encoder pickle file & one hot encoder pickle file
model = load_model('ann_model.h5')


with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

## Streamlit app
st.title('Customer Churn Prediction')
st.write('This is a simple Customer Churn Prediction App')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

if has_cr_card == 'Yes':
    has_cr_card = 1
else:
    has_cr_card = 0

if is_active_member == 'Yes':
    is_active_member = 1
else:
    is_active_member = 0

# Prepare the input data
input_data = pd.DataFrame({'CreditScore': [credit_score],
                           'Gender' : [label_encoder_gender.transform([gender])[0]],
                           'Age': [age],
                            'Tenure': [tenure],
                            'Balance': [balance],
                            'NumOfProducts': [num_of_products],
                            'HasCrCard': [has_cr_card],
                            'IsActiveMember': [is_active_member],
                            'EstimatedSalary': [estimated_salary]})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data, geo_encoded_df], axis = 1)

input_data_scaled = scaler.transform(input_data)

# Predict the output
prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])

if prediction_proba > 0.5:
    st.write('Prediction: Customer will churn')
    st.write('Probability: ', prediction_proba*100, '%')
else:
    st.write('Prediction: Customer will not churn')
    st.write('Probability: ', (1-prediction_proba)*100, '%')
