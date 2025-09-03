🏦 Loan Prediction App

A simple Machine Learning project that predicts whether a loan will be Approved or Rejected based on applicant details.
Built with Python, Scikit-learn, and Streamlit.

🚀 Features

Loan approval prediction using Logistic Regression

Streamlit web app with an interactive form

Real-time prediction with confidence score

Easy to deploy on Streamlit Cloud / Hugging Face

📂 Project Structure
Loan-Prediction/
│── app.py                # Streamlit app for UI
│── train.py              # Model training script
│── loan.csv              # Dataset (sample loan data)
│── loan_approval_model.pkl  # Trained ML model
│── requirements.txt      # Dependencies
│── README.md             # Project documentation

⚙️ Installation

Clone the repository

git clone https://github.com/Gokulm29/Loan-Prediction.git
cd Loan-Prediction


Create virtual environment (optional but recommended)

python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)


Install dependencies

pip install -r requirements.txt

🧑‍💻 Usage
1️⃣ Train Model
python train.py

2️⃣ Run Streamlit App
streamlit run app.py


Then open 👉 http://localhost:8501 in your browser.

🌍 Deployment

Streamlit Cloud: Upload repo and deploy directly

Hugging Face Spaces: Create a space with Streamlit SDK

Render / Railway: Production-ready hosting

📊 Example

Enter details like income, loan amount, dependents, credit history →
Get instant prediction:
✅ Approved or ❌ Rejected

📌 Tech Stack

Python 🐍

Pandas & NumPy

Scikit-learn

Joblib

Streamlit

✨ Future Improvements

Feature importance visualization

Better model (Random Forest, XGBoost)

Export prediction report (CSV/PDF)

👨‍💻 Author

Developed by Gokul Mani 🚀
