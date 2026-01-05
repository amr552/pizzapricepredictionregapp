import streamlit as st
import numpy as np
import joblib
import json

st.set_page_config(page_title="Car Value Estimator", page_icon="🚗")

# --- Custom Styling (Rose Theme) ---
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #4c0519 0%, #881337 100%); }
    .stApp { color: white; font-family: 'Inter', sans-serif; }
    
    /* Force all labels and text to be bold */
    label, .stMarkdown, p, span, div { font-weight: 600 !important; }
    h1, h2, h3 { font-weight: 800 !important; }
    
    .stNumberInput label, .stSelectbox label { font-weight: 700 !important; color: #fecdd3 !important; }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

metadata = json.load(open('metadata.json'))
model = joblib.load('etrcarpricepredict.pkl')

st.title("🚗 AutoValue Pro")
st.write("Get an instant market estimate for any vehicle.")

m = metadata["mappings"]["cars_price"]
c1, c2 = st.columns(2)

with c1:
    brand = st.selectbox("**Brand**", list(m["brand"].keys()))
    year = st.number_input("**Year**", 1990, 2024, 2015)
    km = st.number_input("**Kilometers Driven**", 0, 1000000, 50000)
    fuel = st.selectbox("**Fuel Type**", list(m["fuel"].keys()))
    trans = st.selectbox("**Transmission**", list(m["transmission"].keys()))
    engine = st.number_input("**Engine (CC)**", 600, 7000, 1200)

with c2:
    power = st.number_input("**Max Power (bhp)**", 30, 1000, 80)
    mileage = st.number_input("**Mileage (kmpl)**", 5, 40, 18)
    seats = st.number_input("**Seats**", 2, 14, 5)
    torque = st.number_input("**Torque (Nm)**", 30, 1000, 150)
    owner = st.selectbox("**Owner Type**", list(m["owner"].keys()))
    seller = st.selectbox("**Seller Type**", list(m["seller_type"].keys()))

if st.button("Estimate Price"):
    inputs = [
        m["brand"][brand], year, km, m["fuel"][fuel], m["transmission"][trans],
        engine, power, mileage, seats, torque, m["owner"][owner], m["seller_type"][seller]
    ]
    
    # Model uses log transformation? Based on previous app.py logic
    pred = model.predict(np.array(inputs).reshape(1, -1))[0]
    final_price = np.exp(pred) - 1
    
    st.success(f"Estimated Market Value: ₹ {final_price:,.2f}")
