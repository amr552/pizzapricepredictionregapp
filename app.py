import streamlit as st
import numpy as np
import joblib
import json

st.set_page_config(page_title="Pizza Price Estimator", page_icon="🍕")

# --- Custom Styling (Yellow Theme) ---
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #713f12 0%, #854d0e 100%); }
    .stApp { color: white; font-family: 'Inter', sans-serif; }
    
    /* Force all labels and text to be bold */
    label, .stMarkdown, p, span, div { font-weight: 600 !important; }
    h1, h2, h3 { font-weight: 800 !important; }
    
    .prediction-card {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

metadata = json.load(open('metadata.json'))
model = joblib.load('rfr_pizza_price.pkl')
scaler = joblib.load('pizzaregscaler.joblib')

st.title("🍕 PizzaPrice Genius")
st.write("Calculate the perfect price for your custom pizza.")

m = metadata["mappings"]["pizza_price"]
c1, c2 = st.columns(2)

with c1:
    company = st.selectbox("**Company**", list(m["company"].keys()))
    diameter = st.number_input("**Diameter (Inches)**", 8.0, 30.0, 12.0)
    topping = st.selectbox("**Topping**", list(m["topping"].keys()))
    variant = st.selectbox("**Variant**", list(m["variant"].keys()))

with c2:
    size = st.selectbox("**Size**", list(m["size"].keys()))
    sauce = st.selectbox("**Extra Sauce?**", ["yes", "no"])
    cheese = st.selectbox("**Extra Cheese?**", ["yes", "no"])
    mushrooms = st.selectbox("**Extra Mushrooms?**", ["yes", "no"])

if st.button("Calculate Price"):
    # Scaler expects only 1 feature (diameter)
    scaled_diameter = scaler.transform(np.array([[diameter]]))[0][0]
    
    inputs = [
        m["company"][company], scaled_diameter, m["topping"][topping], 
        m["variant"][variant], m["size"][size],
        1 if sauce=='yes' else 0, 1 if cheese=='yes' else 0, 1 if mushrooms=='yes' else 0
    ]
    
    pred = model.predict(np.array(inputs).reshape(1, -1))[0]
    price = np.exp(pred) - 1
    
    st.success(f"Estimated Pizza Price: ₹ {price:,.2f}")
