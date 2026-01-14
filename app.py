import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('logistic_regression_model.sav', 'rb'))

st.title('Credit Card Fraud Detection')
st.write('Enter the details to predict if the transaction is fraudulent or legitimate.')

# Create input fields for the features
# Based on the dataset's columns (excluding 'Time' and 'Class')
# For simplicity, we'll create generic V1-V28, Amount inputs.
# In a real-world scenario, you might want to standardize or normalize inputs if the model was trained on scaled data.

input_features = {}
for i in range(1, 29):
    input_features[f'V{i}'] = st.number_input(f'Enter V{i} value', value=0.0, format="%.6f")

input_features['Amount'] = st.number_input('Enter Amount value', value=0.0, format="%.2f")

# Adding a 'Time' feature, assuming it's also a relevant input
input_features['Time'] = st.number_input('Enter Time value', value=0.0, format="%.0f")

# Convert input to a numpy array for prediction
# Ensure the order of features matches the training data
feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time'] # This order might need adjustment based on X_train columns

# Re-order feature_names to match the original X DataFrame if 'Time' is at the beginning
original_X_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Check if all keys exist in input_features before creating the array
if all(key in input_features for key in original_X_columns):
    input_data = [input_features[col] for col in original_X_columns]
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    if st.button('Predict'):
        prediction = model.predict(input_data_as_numpy_array)

        if prediction[0] == 0:
            st.success('Legitimate Transaction')
        else:
            st.error('Fraudulent Transaction')
else:
    st.warning("Some input features are missing. Please provide all required inputs.")
