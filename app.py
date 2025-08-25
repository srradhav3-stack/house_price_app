import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r"C:\Users\Srradha\Downloads\house_prices.csv")


# Select relevant columns
X = df[["bedrooms", "bathrooms", "sqft_living"]]
y = df["price"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏠 House Price Prediction App")
st.write("Enter house details to predict price:")

# User input sliders
bedrooms = st.slider("Number of Bedrooms", int(X["bedrooms"].min()), int(X["bedrooms"].max()), 3)
bathrooms = st.slider("Number of Bathrooms", float(X["bathrooms"].min()), float(X["bathrooms"].max()), 2.0)
sqft_living = st.slider("Living Area (sqft)", int(X["sqft_living"].min()), int(X["sqft_living"].max()), 1500)

# Prediction
if st.button("Predict Price"):
    features = [[bedrooms, bathrooms, sqft_living]]
    predicted_price = model.predict(features)[0]
    st.success(f"💰 Estimated House Price: ${predicted_price:,.2f}")
 