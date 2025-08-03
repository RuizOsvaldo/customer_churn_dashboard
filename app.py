# customer_churn_dashboard.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load data
try:
    df = pd.read_csv("customer_churn.csv")
    if df.empty:
        st.error("Dataset is empty")
        st.stop()
except FileNotFoundError:
    st.error("customer_churn.csv file not found")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

st.title("Customer Churn Analysis Dashboard")

# Display raw data with expander
with st.expander("Show Raw Data"):
    st.dataframe(df)

# Data preprocessing
df = df.dropna()
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Feature encoding
df_encoded = pd.get_dummies(df.drop("customerID", axis=1), drop_first=True)
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
report = classification_report(y_test, preds, output_dict=True)
conf_matrix = confusion_matrix(y_test, preds)

st.subheader("Model Performance")
st.markdown("**Classification Report**")
st.dataframe(pd.DataFrame(report).transpose().round(2))

st.markdown("**Confusion Matrix**")
st.dataframe(pd.DataFrame(conf_matrix, columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"]))

# Feature importance
st.subheader("Top 10 Features Influencing Churn")
feat_importance = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
clean_names = {
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
    "SeniorCitizen": "Senior Citizen"
}
feat_importance.index = [clean_names.get(col, col.replace("_", " ").title()) for col in feat_importance.index]
fig, ax = plt.subplots()
sns.barplot(x=feat_importance, y=feat_importance.index, ax=ax)
ax.set_title("Top 10 Features Influencing Churn")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
st.pyplot(fig)
