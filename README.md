# 🏠 House Price Prediction – End-to-End Machine Learning Project

This project demonstrates an **end-to-end Machine Learning workflow** for predicting house prices based on multiple property features such as location, size, BHK, amenities, and accessibility.  
The final trained model is deployed as an interactive **Streamlit web application** and also hosted on **Hugging Face Spaces**.

---

## 📌 Problem Statement
Accurately estimating house prices is essential for buyers, sellers, and real-estate platforms. Manual pricing approaches are often subjective and inconsistent.  
This project aims to build a **data-driven house price prediction system** using machine learning techniques to provide reliable price estimates.

---

## 🧠 Project Workflow
1. Data understanding and cleaning  
2. Exploratory Data Analysis (EDA)  
3. Feature selection and engineering  
4. Model training and evaluation  
5. Model deployment using Streamlit  

---

## 📊 Dataset
- Real-world–inspired housing dataset  
- 50,000+ records  
- 23 columns including:
  - State, City  
  - Property Type, BHK, Size (SqFt)  
  - Furnishing Status, Amenities  
  - Nearby Schools & Hospitals  
  - Public Transport Accessibility  
  - Parking & Security  
  - Price in Lakhs (target variable)

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Machine Learning Models:**  
  - Decision Tree Regressor  
  - KNN Regressor  
- **Deployment Platforms:**  
  - Streamlit  
  - Hugging Face Spaces  
- **Version Control:** Git & GitHub  

---

## 📈 Model Evaluation
The models were evaluated using standard regression metrics:
- **R² Score**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

The final model demonstrated strong generalization performance on unseen test data.

---

## 🌐 Live Applications

### 🔹 Streamlit Deployment
👉 https://housepricepredictionmlproject-leaeeojfchrmnqgwwmqm4e.streamlit.app/

### 🔹 Hugging Face Spaces Deployment
👉 https://huggingface.co/spaces/nithinpallapu/House_Price_Prediction

Both deployments provide an interactive UI where users can input property details and receive predicted house prices in real time.
