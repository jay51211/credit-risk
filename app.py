import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

st.title("ML-Based Credit Score Predictor")
st.subheader("Predict creditworthiness using machine learning")
st.caption("Open side bar to fill information of person.")

@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()

categorical_cols = df.select_dtypes(include = ["object"]).columns
numerical_cols = df.select_dtypes(include = ["float64"]).columns
nums_imp = SimpleImputer(strategy = "mean")
df[numerical_cols] = nums_imp.fit_transform(df[numerical_cols])
cat_imp = SimpleImputer(strategy = "most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

df = df.drop(columns = "Applicant_ID")
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]

ohe = OneHotEncoder(drop = "first", sparse_output = False, handle_unknown = "ignore")

encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(encoded, columns = ohe.get_feature_names_out(cols), index = df.index)

df = pd.concat([df.drop(columns = cols), encoded_df], axis = 1)

df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

X = df.drop(columns = ["DTI_Ratio", "Credit_Score", "Loan_Approved", "Applicant_Income"])
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

naive_model = GaussianNB()
naive_model.fit(X_train_scaled, y_train)

y_pred = naive_model.predict(X_test_scaled)

st.sidebar.header("Enter customer details")
user_input = {}
st.sidebar.subheader("Select 1 if the condition applies to you; otherwise, select 0.")

check_col = [""]

for col in X.columns:
    label = col.replace("_", " ")

    user_input[col] = st.sidebar.number_input(
        label,
        min_value=int(df[col].min()),
        max_value=int(df[col].max()),
        value=int(df[col].mean())
    )

input_df = pd.DataFrame([user_input])
input_scaled =scaler.transform(input_df)

if st.button("Predict"):
    prediction = naive_model.predict(input_scaled)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Good Credit Score")
    else:
        st.error("Poor Credit Score")


st.markdown("---")
st.caption("⚠️ This app is for educational purposes only.")
