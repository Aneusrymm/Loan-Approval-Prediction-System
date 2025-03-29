import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Set page config
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="wide")

# Title and description
st.title("Loan Approval Prediction System")
st.markdown("""
This app predicts whether a loan application will be approved based on applicant information.
Select a model from the sidebar and fill in the applicant details.
""")

# Load available models
@st.cache_data
def load_available_models():
    models_dir = "models"
    available_models = {}
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(('.pkl', '.joblib', '.sav')):
                model_name = os.path.splitext(file)[0]
                available_models[model_name] = os.path.join(models_dir, file)
    
    return available_models

available_models = load_available_models()

# Sidebar for model selection and user inputs
# ... (keep all your imports and initial setup code the same) ...

# Load available models
@st.cache_data
def load_available_models():
    models_dir = "models"
    available_models = {}
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(('.pkl', '.joblib', '.sav')):
                model_name = os.path.splitext(file)[0]
                available_models[model_name] = os.path.join(models_dir, file)
    
    return available_models

available_models = load_available_models()

# Sidebar for model selection and user inputs
with st.sidebar:
    st.header("Model Selection")
    
    if not available_models:
        st.warning("No models found in the 'models' folder. Please add trained models.")
        st.stop()
    
    selected_model_name = st.selectbox("Choose a model", list(available_models.keys()))
    
    st.header("Applicant Information")
    
    # User inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    # Slider inputs for numerical values
    applicant_income = st.slider(
        "Applicant Income ($)",
        min_value=0,
        max_value=50000,
        value=5000,
        step=100
    )
    
    coapplicant_income = st.slider(
        "Coapplicant Income ($)",
        min_value=0,
        max_value=30000,
        value=2000,
        step=100
    )
    
    loan_amount = st.slider(
        "Loan Amount ($)",
        min_value=0,
        max_value=700,
        value=100,
        step=10
    )
    
    loan_amount_term = st.slider(
        "Loan Term (days)",
        min_value=0,
        max_value=480,
        value=360,
        step=30
    )
    
    credit_history = st.selectbox("Credit History", [1.0, 0.0], 
                                format_func=lambda x: "Good" if x == 1.0 else "Bad")
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Load the selected model AFTER the sidebar section
@st.cache_resource
def load_model(model_path):
    try:
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Now load the model after the sidebar inputs are defined
model = load_model(available_models[selected_model_name])

# ... (rest of your code remains the same) ...

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('Dataset.csv')
    
    # Fill missing values
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
    
    # Fill categorical missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert categorical to numerical
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Feature engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['ApplicantIncomelog'] = np.log(df['ApplicantIncome'])
    df['LoanAmountlog'] = np.log(df['LoanAmount'])
    df['Loan_Amount_Term_log'] = np.log(df['Loan_Amount_Term'])
    df['Total_Income_log'] = np.log(df['Total_Income'])
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['Dependents', 'Education', 'Self_Employed', 'Property_Area'])
    
    return df

df_processed = load_and_preprocess_data()

# Separate features and target
X = df_processed.drop(['Loan_ID', 'Loan_Status', 'ApplicantIncome', 'LoanAmount', 
                      'Loan_Amount_Term', 'Total_Income'], axis=1)
y = df_processed['Loan_Status']

# Apply oversampling
oversample = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversample.fit_resample(X, y)

# Create resampled DataFrame for visualization
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                         pd.Series(y_resampled, name="Loan_Status")], axis=1)

# Split data (for visualization only)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                   test_size=0.2, 
                                                   random_state=42)

# Update the prepare_input_data function to match exactly what the model expects
def prepare_input_data(user_input):
    """Prepare user input for prediction with proper feature engineering and validation"""
    try:
        # Create initial DataFrame with input validation
        input_df = pd.DataFrame({
            'Gender': [1 if user_input['gender'] == "Male" else 0],
            'Married': [1 if user_input['married'] == "Yes" else 0],
            'Dependents': [user_input['dependents']],
            'Education': [1 if user_input['education'] == "Graduate" else 0],
            'Self_Employed': [1 if user_input['self_employed'] == "Yes" else 0],
            'ApplicantIncome': [max(1, user_input['applicant_income'])],  # Ensure at least 1
            'CoapplicantIncome': [max(0, user_input['coapplicant_income'])],  # Ensure non-negative
            'LoanAmount': [max(1, user_input['loan_amount'])],  # Ensure at least 1
            'Loan_Amount_Term': [max(1, user_input['loan_amount_term'])],  # Ensure at least 1
            'Credit_History': [user_input['credit_history']],
            'Property_Area': [user_input['property_area']]
        })
        
        # Apply transformations with safety checks
        def safe_log(x):
            return np.log(max(1, x))  # Ensure we never take log(0) or log(negative)
        
        input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
        input_df['ApplicantIncomelog'] = input_df['ApplicantIncome'].apply(safe_log)
        input_df['LoanAmountlog'] = input_df['LoanAmount'].apply(safe_log)
        input_df['Loan_Amount_Term_log'] = input_df['Loan_Amount_Term'].apply(safe_log)
        input_df['Total_Income_log'] = input_df['Total_Income'].apply(safe_log)
        
        # One-hot encode categorical variables
        dependents_mapping = {'0': [1, 0, 0, 0], 
                            '1': [0, 1, 0, 0],
                            '2': [0, 0, 1, 0],
                            '3+': [0, 0, 0, 1]}
        dep_values = dependents_mapping[user_input['dependents']]
        input_df['Dependents_0'] = dep_values[0]
        input_df['Dependents_1'] = dep_values[1]
        input_df['Dependents_2'] = dep_values[2]
        input_df['Dependents_3+'] = dep_values[3]
        
        property_mapping = {'Urban': [0, 0, 1],
                          'Rural': [1, 0, 0],
                          'Semiurban': [0, 1, 0]}
        prop_values = property_mapping[user_input['property_area']]
        input_df['Property_Area_Rural'] = prop_values[0]
        input_df['Property_Area_Semiurban'] = prop_values[1]
        input_df['Property_Area_Urban'] = prop_values[2]
        
        # Drop original columns
        input_df = input_df.drop(['Dependents', 'Property_Area'], axis=1)
        
        # Get expected columns (from model or our knowledge)
        if hasattr(model, 'feature_names_in_'):
            expected_columns = list(model.feature_names_in_)
        else:
            expected_columns = [
                'Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History',
                'ApplicantIncomelog', 'LoanAmountlog', 'Loan_Amount_Term_log', 'Total_Income_log',
                'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
                'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
            ]
        
        # Add any missing columns with 0
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns
        input_df = input_df[expected_columns]
        
        # Final check for infinite or extremely large values
        if np.isinf(input_df.values).any() or np.isnan(input_df.values).any():
            st.error("Invalid numerical values detected in input data")
            return None
            
        return input_df
        
    except Exception as e:
        st.error(f"Error preparing input data: {str(e)}")
        return None

# In your prediction code:
if st.sidebar.button("Predict Loan Approval"):
    if model is None:
        st.error("Model failed to load. Please check the model file.")
    else:
        user_input = {
            'gender': gender,
            'married': married,
            'dependents': dependents,
            'education': education,
            'self_employed': self_employed,
            'applicant_income': applicant_income,
            'coapplicant_income': coapplicant_income,
            'loan_amount': loan_amount,
            'loan_amount_term': loan_amount_term,
            'credit_history': credit_history,
            'property_area': property_area
        }
        
        input_data = prepare_input_data(user_input)
        
        if input_data is not None:
            try:
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)
                
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.success(f"✅ Loan Approved (using {selected_model_name})")
                else:
                    st.error(f"❌ Loan Not Approved (using {selected_model_name})")
                
                st.write(f"Confidence: {prediction_proba[0][prediction[0]]*100:.2f}%")
                
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': input_data.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
                    ax.set_title('Top 10 Important Features')
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
# Model information section
st.header("Model Information")
st.write(f"Currently selected model: **{selected_model_name}**")

# Show model type
if model is not None:
    st.write(f"Model type: {type(model).__name__}")

# Data exploration section
st.header("Data Exploration")

# Show class distribution before and after resampling
st.subheader("Class Distribution")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(x=y, ax=ax1)
    ax1.set_title('Original Class Distribution')
    ax1.set_xticklabels(['Rejected', 'Approved'])
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x=y_resampled, ax=ax2)
    ax2.set_title('After Oversampling')
    ax2.set_xticklabels(['Rejected', 'Approved'])
    st.pyplot(fig2)

# Show raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df_processed)  # Use df_processed instead of df

# Show data statistics
if st.checkbox("Show data statistics"):
    st.subheader("Data Statistics")
    st.write(df_processed.describe())  # Use df_processed instead of df

# Visualizations
st.subheader("Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Gender distribution
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Gender', data=df_processed, palette='Set1', ax=ax1)
    ax1.set_title('Number of Applicants by Gender')
    st.pyplot(fig1)

with col2:
    # Married status distribution
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Married', data=df_processed, palette='Set2', ax=ax2)
    ax2.set_title('Number of Applicants by Marital Status')
    st.pyplot(fig2)

# Loan status by property area
# Note: Property_Area columns are now one-hot encoded, so we need to handle this differently
# We'll need to reconstruct the original Property_Area column for visualization
if 'Property_Area_Rural' in df_processed.columns:
    df_processed['Property_Area'] = np.where(df_processed['Property_Area_Rural'] == 1, 'Rural',
                                          np.where(df_processed['Property_Area_Semiurban'] == 1, 'Semiurban', 'Urban'))

fig3, ax3 = plt.subplots(figsize=(10, 4))
sns.countplot(x='Property_Area', hue='Loan_Status', data=df_processed, ax=ax3)
ax3.set_title('Loan Status by Property Area')
st.pyplot(fig3)