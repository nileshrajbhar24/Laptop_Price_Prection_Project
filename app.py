import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Load data from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/nileshrajbhar24/Laptop_Price_Prection_Project/refs/heads/main/cleaned_laptop_prices.csv"
    return pd.read_csv(url)

# Train a simple model with ALL required features
@st.cache_resource
def train_model(df):
    # Create a simple model with all required features
    df_model = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Company', 'TypeName', 'CPU_company', 'GPU_company', 'OS']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le
    
    # Select features and target - include ALL features the model needs
    features = ['Company', 'TypeName', 'Inches', 'Ram', 'Weight', 'CPU_company', 'GPU_company', 'PrimaryStorage']
    X = df_model[features]
    y = df_model['Price_euros']
    
    # Train the model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, label_encoders

# Load data and train model
df = load_data()
model, label_encoders = train_model(df)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a page", ["Price Prediction", "Data Exploration", "About"])

if app_mode == "Price Prediction":
    st.title('💻 Laptop Price Predictor')
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Specifications")
        company = st.selectbox('Brand', sorted(df['Company'].unique()))
        type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
        inches = st.slider('Screen Size (inches)', min_value=10.0, max_value=18.0, value=15.6, step=0.1)
        ram = st.select_slider('RAM (GB)', options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
        weight = st.slider('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        
    with col2:
        st.subheader("Advanced Specifications")
        cpu = st.selectbox('CPU Brand', sorted(df['CPU_company'].unique()))
        gpu = st.selectbox('GPU Brand', sorted(df['GPU_company'].unique()))
        storage = st.select_slider('Storage (GB)', options=[32, 64, 128, 256, 512, 1024, 2048], value=256)
        os = st.selectbox('Operating System', sorted(df['OS'].unique()))
    
    # EURO TO RUPEE CONVERSION RATE
    EURO_TO_RUPEE_RATE = 90.0
    
    if st.button('Predict Price', type="primary"):
        try:
            # Prepare input data with ALL required features
            input_data = pd.DataFrame({
                'Company': [company],
                'TypeName': [type_name],
                'Inches': [inches],
                'Ram': [ram],
                'Weight': [weight],
                'CPU_company': [cpu],
                'GPU_company': [gpu],  # This was missing before
                'PrimaryStorage': [storage]  # This was missing before
            })
            
            # Encode categorical variables
            input_data_encoded = input_data.copy()
            for col in ['Company', 'TypeName', 'CPU_company', 'GPU_company']:
                input_data_encoded[col] = label_encoders[col].transform(input_data_encoded[col])
            
            # Make prediction
            prediction = model.predict(input_data_encoded)
            euro_price = prediction[0]
            rupee_price = euro_price * EURO_TO_RUPEE_RATE
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"### Predicted Price (Euros)\n€{euro_price:,.2f}")
            with col2:
                st.success(f"### Predicted Price (Rupees)\n₹{rupee_price:,.2f}")
            
            st.info(f"*Conversion rate: 1€ = ₹{EURO_TO_RUPEE_RATE}*")
            
            # Show similar laptops
            st.subheader("Similar Laptops")
            similar = df[
                (df['Company'] == company) & 
                (df['TypeName'] == type_name) & 
                (df['Ram'] == ram)
            ].head(5)
            
            if not similar.empty:
                similar['Price_rupees'] = similar['Price_euros'] * EURO_TO_RUPEE_RATE
                st.dataframe(similar[['Company', 'TypeName', 'Ram', 'Inches', 'Price_euros', 'Price_rupees']])
            else:
                st.info("No similar laptops found in our database.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif app_mode == "Data Exploration":
    st.title('📊 Laptop Data Exploration')
    
    st.subheader("Dataset Overview")
    st.write(f"Total laptops in dataset: {len(df)}")
    
    # Show raw data with filters
    st.subheader("Filter Data")
    col1, col2 = st.columns(2)
    with col1:
        min_price, max_price = st.slider(
            "Price Range (€)",
            min_value=int(df['Price_euros'].min()),
            max_value=int(df['Price_euros'].max()),
            value=(int(df['Price_euros'].min()), int(df['Price_euros'].max()))
        )
        selected_brands = st.multiselect(
            "Select Brands",
            options=df['Company'].unique(),
            default=df['Company'].unique()[:3]
        )
    with col2:
        min_ram, max_ram = st.slider(
            "RAM Range (GB)",
            min_value=int(df['Ram'].min()),
            max_value=int(df['Ram'].max()),
            value=(int(df['Ram'].min()), int(df['Ram'].max()))
        )
    
    filtered_df = df[
        (df['Price_euros'].between(min_price, max_price)) &
        (df['Company'].isin(selected_brands)) &
        (df['Ram'].between(min_ram, max_ram))
    ]
    
    st.write(f"Showing {len(filtered_df)} laptops")
    st.dataframe(filtered_df.head(20))
    
    # Visualizations
    st.subheader("Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Brand Comparison", "Feature Correlations"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df['Price_euros'], bins=20, kde=True, ax=ax)
        ax.set_title('Price Distribution')
        ax.set_xlabel('Price (€)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 6))
        brand_avg = filtered_df.groupby('Company')['Price_euros'].mean().sort_values(ascending=False)
        sns.barplot(x=brand_avg.index, y=brand_avg.values, ax=ax)
        ax.set_title('Average Price by Brand')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        
    with tab3:
        # Select numeric columns for correlation
        numeric_cols = ['Price_euros', 'Ram', 'Inches', 'Weight', 'PrimaryStorage']
        numeric_df = filtered_df[numeric_cols].dropna()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap='coolwarm',
            ax=ax
        )
        ax.set_title('Feature Correlations')
        st.pyplot(fig)

elif app_mode == "About":
    st.title('ℹ️ About This Project')
    
    st.markdown("""
    ## Laptop Price Predictor
    
    This application predicts laptop prices based on their specifications using machine learning.
    
    ### Features:
    - **Price Prediction**: Estimate laptop prices based on specifications
    - **Data Exploration**: Explore the dataset with filters and visualizations
    - **Similar Products**: Find similar laptops in our database
    
    ### How It Works:
    1. Select your desired laptop specifications
    2. Click "Predict Price" to get an estimated price
    3. Explore similar laptops in our database
    
    ### Data Source:
    The dataset contains information about various laptop models with their specifications and prices.
    
    ### Model Information:
    - Algorithm: Random Forest Regressor
    - Features used: Brand, Type, Screen Size, RAM, Weight, CPU, GPU, Storage
    """)
    
    st.markdown("---")
    st.markdown("Created with ❤️ using Python and Streamlit")
