import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- PAGE SETUP ---
st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide")

st.title("🤖 AI-Powered Market Intelligence Dashboard")
st.markdown("### Analyzing Price Anomalies in Pakistani Hardware Market")


# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    # Attempt to load the pre-calculated report
    try:
        df = pd.read_csv('final_market_report.csv')
        return df
    except FileNotFoundError:
        # Fallback: Load processed data and retrain (simplified for demo)
        try:
            df = pd.read_csv('processed_data_final.csv')
            # Basic training logic if pre-computed report is missing
            # (Ideally, you run the ML script first to generate the report)
            return df
        except:
            return pd.DataFrame()


df = load_data()

if df.empty:
    st.error(
        "❌ Data not found! Please ensure 'final_market_report.csv' or 'processed_data_final.csv' is in the project folder.")
    st.stop()

# Ensure Scalp_Score exists (if using fallback)
if 'Scalp_Score' not in df.columns:
    st.warning("⚠️ Pre-calculated scores missing. Please run the ML analysis script first for full details.")
    df['Scalp_Score'] = 0  # Placeholder
    df['Fair_Price'] = df['Price_PKR']  # Placeholder

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Options")
selected_category = st.sidebar.selectbox("Category", ["All"] + list(df['Category'].unique()))
selected_brand = st.sidebar.multiselect("Brand", df['Brand'].unique(), default=df['Brand'].unique())

# Apply Filters
filtered_df = df[df['Brand'].isin(selected_brand)]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]

# --- 2. KEY METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Items", len(filtered_df))
col2.metric("Avg Price", f"PKR {int(filtered_df['Price_PKR'].mean()):,}")
scalpers = filtered_df[filtered_df['Scalp_Score'] > 30]
deals = filtered_df[filtered_df['Scalp_Score'] < -30]
col3.metric("Scalpers Detected", len(scalpers), delta="-High Risk", delta_color="inverse")
col4.metric("Potential Deals", len(deals), delta="Good Value")

st.divider()

# --- 3. TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Market Visuals", "🚨 Scalper Detector", "🔮 Price Calculator"])

with tab1:
    st.subheader("Market Landscape")

    # Scatter Plot: Specs vs Price
    # Dynamic X-axis based on category
    x_axis = 'VRAM' if selected_category == 'GPU' else 'RAM'

    if x_axis in filtered_df.columns:
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y='Price_PKR',
            color='Brand',
            size='Price_PKR',
            hover_data=['Product Name', 'Fair_Price'],
            title=f"Price vs. {x_axis} Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select a specific category (Laptop or GPU) to see detailed spec comparisons.")

with tab2:
    st.subheader("🚩 Detected Price Anomalies")
    st.markdown("Items priced **>30% above** the AI-predicted fair value.")

    # Display Scalpers Table
    if not scalpers.empty:
        st.dataframe(
            scalpers[['Product Name', 'Category', 'Price_PKR', 'Fair_Price', 'Scalp_Score']].sort_values('Scalp_Score',
                                                                                                         ascending=False),
            column_config={
                "Price_PKR": st.column_config.NumberColumn(format="PKR %d"),
                "Fair_Price": st.column_config.NumberColumn(format="PKR %d"),
                "Scalp_Score": st.column_config.ProgressColumn("Overpriced %", format="%.1f%%", min_value=0,
                                                               max_value=100)
            },
            use_container_width=True
        )
    else:
        st.success("No major scalpers detected in this selection.")

with tab3:
    st.subheader("Predict Fair Value")
    st.markdown("Enter specs to get an AI estimated price.")

    # Simple Input Form
    c1, c2, c3 = st.columns(3)
    in_ram = c1.number_input("RAM (GB)", 4, 128, 16)
    in_storage = c2.number_input("Storage (GB)", 256, 4096, 512)
    in_vram = c3.number_input("VRAM (GB) [GPU Only]", 0, 24, 8)

    if st.button("Calculate Fair Price"):
        # Dummy prediction logic for the prototype (replace with real model inference if needed)
        # This formula mimics a basic linear weight for demonstration
        base_price = 50000
        ram_price = in_ram * 4000
        storage_price = in_storage * 20
        vram_price = in_vram * 15000

        predicted_val = base_price + ram_price + storage_price + vram_price
        st.metric(label="Estimated Fair Market Price", value=f"PKR {predicted_val:,}")
        st.caption("*Estimation based on market averages.")