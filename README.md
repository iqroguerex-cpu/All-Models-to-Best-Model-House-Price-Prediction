# 🏠 ProphetPrice — Real Estate Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen?logo=streamlit)](https://atnof3tj3mfeujkk4ghwhm.streamlit.app/)

---

## 📌 Overview

**ProphetPrice** is an end-to-end Machine Learning web application that predicts real estate prices using multiple regression algorithms and provides interactive analytics through a modern Streamlit dashboard.

It combines:

* 📊 Data analytics
* 🤖 Machine learning models
* 📈 Interactive visualizations
* 🧠 Automated model selection

---

## ✨ Features

* 🔍 **Multiple Regression Models**

  * Linear Regression
  * Support Vector Regression (SVR)
  * Decision Tree
  * Random Forest

* 📊 **Model Performance Dashboard**

  * MAE, RMSE, R² comparison
  * Automatic best model selection

* 📈 **Interactive Visualizations**

  * Feature importance (Random Forest)
  * Actual vs Predicted scatter plots
  * Model comparison charts

* 🧠 **Smart Prediction Engine**

  * Uses best-performing model dynamically
  * Real-time user input predictions

* ⚡ **Optimized Performance**

  * Caching with `st.cache_data` and `st.cache_resource`

---

## 🧠 Machine Learning Workflow

Data → Preprocessing → Scaling → Model Training → Evaluation → Visualization → Prediction

---

## 📊 Evaluation Metrics

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score (Model Accuracy)**

---

## 🛠️ Tech Stack

| Category      | Tools Used    |
| ------------- | ------------- |
| Language      | Python 🐍     |
| ML Library    | Scikit-learn  |
| Visualization | Plotly        |
| Web Framework | Streamlit     |
| Data Handling | Pandas, NumPy |

---

## 📂 Project Structure

```
ProphetPrice
│
├── app.py
├── house_price_dataset.csv
├── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/prophetprice.git
cd prophetprice
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run App

```
streamlit run app.py
```

---

## 🎯 Key Highlights

* 🔥 Clean ML pipeline implementation
* 📊 Professional dashboard UI
* ⚡ Real-time predictions
* 🧠 Automated best model selection
* 📈 Business-ready analytics

---

## 🚀 Future Improvements

* Add Polynomial Regression
* Hyperparameter tuning (GridSearchCV)
* Docker deployment
* Database integration
* Time-series forecasting

---

## 👨‍💻 Author

Chinmay V Chatradamath
---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it really helps!
