# main.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image

# Load saved model and scaler
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
options = ["Home", "About", "Evaluation", "Disease Prediction"]
choice = st.sidebar.radio("Go to", options)

# Home Page
if choice == "Home":
    st.title("🧠 Stroke Prediction System")
    st.markdown("""
        Welcome to the **Stroke Prediction System**.  
        This application predicts the likelihood of stroke based on patient health data.  
        Use the sidebar to navigate through the app.
    """)
    
    # Add an image
    image = Image.open("stroke_image.jpg")  # Add an image in your project folder
    st.image(image, use_container_width =True)
    
    st.subheader("How it works:")
    st.markdown("""
    1. Input your health parameters in the **Disease Prediction** section.  
    2. The system uses a trained Random Forest model to predict stroke risk.  
    3. Review the risk assessment and take preventive measures if needed.
    """)
    
    st.subheader("Example Patients:")
    example_data = pd.DataFrame({
        "Gender": ["Male","Female","Female"],
        "Age": [45, 60, 30],
        "Hypertension": [0,1,0],
        "Heart Disease": [0,1,0],
        "Ever Married": ["Yes","Yes","No"],
        "Work Type": ["Private","Govt_job","Self-employed"],
        "Residence": ["Urban","Rural","Urban"],
        "Glucose": [100, 180, 90],
        "BMI": [25, 30, 22],
        "Smoking Status": ["never smoked","formerly smoked","smokes"]
    })
    st.dataframe(example_data)

# About Page
elif choice == "About":
    st.title("ℹ️ About Stroke Prediction")
    
    # Add an image
    about_image = Image.open("brain_health.jpg")  # Add an image in your project folder
    st.image(about_image, use_container_width=True)
    
    st.markdown("""
        **Stroke** is a medical condition where poor blood flow to the brain results in cell death.  
        Early detection is crucial to prevent severe health consequences.
    """)
    
    st.subheader("Why use this system?")
    st.markdown("""
    - **Early Awareness:** Helps identify high-risk individuals.  
    - **Data-Driven:** Uses machine learning (Random Forest) for predictions.  
    - **User-Friendly:** Easy-to-use web interface with clear guidance.  
    - **Preventive Measures:** Encourages timely medical consultation.
    """)

# Evaluation Page
elif choice == "Evaluation":
    st.title("Model Evaluation Metrics")
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load the dataset for evaluation
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df['bmi'].fillna(df['bmi'].median(), inplace=True)
    label_enc_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in label_enc_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    X = df.drop(['id','stroke'], axis=1)
    y = df['stroke']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    
    st.write("**Accuracy:**", np.round(accuracy_score(y_test, y_pred),4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# Disease Prediction Page
elif choice == "Disease Prediction":
    st.title("Stroke Prediction Form")
    
    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", 0, 120, 50)
    hypertension = st.selectbox("Hypertension", [0,1])
    heart_disease = st.selectbox("Heart Disease", [0,1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private","Self-employed","Govt_job","children","Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban","Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked","never smoked","smokes","Unknown"])
    
    # Encode inputs
    input_df = pd.DataFrame({
        "gender":[0 if gender=="Male" else 1 if gender=="Female" else 2],
        "age":[age],
        "hypertension":[hypertension],
        "heart_disease":[heart_disease],
        "ever_married":[0 if ever_married=="No" else 1],
        "work_type":[["Private","Self-employed","Govt_job","children","Never_worked"].index(work_type)],
        "Residence_type":[0 if Residence_type=="Urban" else 1],
        "avg_glucose_level":[avg_glucose_level],
        "bmi":[bmi],
        "smoking_status":[["formerly smoked","never smoked","smokes","Unknown"].index(smoking_status)]
    })
    
    input_scaled = scaler.transform(input_df)
    
    if st.button("Predict"):
        prediction = model.predict(input_scaled)[0]
        if prediction==1:
            st.error("⚠️ High risk of stroke. Consult a doctor!")
        else:
            st.success("✅ Low risk of stroke.")
