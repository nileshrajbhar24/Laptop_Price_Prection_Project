import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Set page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Load data from GitHub
@st.cache_data
def load_data():
    # Use GitHub raw URL instead of local path
    url = "https://raw.githubusercontent.com/nileshrajbhar24/Laptop_Price_Prection_Project/refs/heads/main/cleaned_laptop_prices.csv"
    return pd.read_csv(url)

# Load model from GitHub
@st.cache_resource
def load_model():
    try:
        # Model URLs - use the exact URLs you copied from GitHub
        model_url = "https://github.com/nileshrajbhar24/Laptop_Price_Prection_Project/raw/main/laptop_model1.pkl"
        label_encoders_url = "https://github.com/nileshrajbhar24/Laptop_Price_Prection_Project/raw/main/label_encoders1.pkl"
        
        # Download model files
        model_response = requests.get(model_url)
        label_encoders_response = requests.get(label_encoders_url)
        
        # Check if files were found
        if model_response.status_code != 200:
            st.error(f"Model file not found at: {model_url}")
            raise FileNotFoundError("Model file not found")
            
        if label_encoders_response.status_code != 200:
            st.error(f"Label encoders file not found at: {label_encoders_url}")
            raise FileNotFoundError("Label encoders file not found")
        
        # Load from downloaded content
        model = pickle.load(BytesIO(model_response.content))
        label_encoders = pickle.load(BytesIO(label_encoders_response.content))
        
        return model, label_encoders
        
    except Exception as e:
        st.warning(f"Could not load model files: {str(e)}. Training a new model with your data...")
        
        # Train a new model
        df = load_data()
        
        # Preprocess the data
        df_model = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['Company', 'TypeName', 'CPU_company', 'GPU_company', 'OS']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le
        
        # Select features and target
        features = ['Company', 'TypeName', 'Inches', 'Ram', 'Weight', 'CPU_company']
        X = df_model[features]
        y = df_model['Price_euros']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.success(f"New model trained successfully! MAE: €{mae:.2f}, R²: {r2:.3f}")
        
        return model, label_encoders

df = load_data()
model, label_encoders = load_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a page", ["Price Prediction", "Data Exploration", "About"])

if app_mode == "Price Prediction":
    # Main prediction interface
    st.title('💻 Laptop Price Predictor')
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic specifications
        st.subheader("Basic Specifications")
        company = st.selectbox('Brand', sorted(df['Company'].unique()))
        type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
        inches = st.slider('Screen Size (inches)', min_value=10.0, max_value=18.0, value=15.6, step=0.1)
        ram = st.select_slider('RAM (GB)', options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
        weight = st.slider('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        
    with col2:
        # Advanced specifications
        st.subheader("Advanced Specifications")
        cpu = st.selectbox('CPU Brand', sorted(df['CPU_company'].unique()))
        cpu_model = st.text_input('CPU Model (e.g., Core i5, Ryzen 7)', 'Core i5')
        storage = st.select_slider('Primary Storage (GB)', options=[32, 64, 128, 256, 512, 1024, 2048], value=256)
        storage_type = st.selectbox('Storage Type', ['SSD', 'HDD', 'Flash Storage', 'Hybrid'])
        gpu = st.selectbox('GPU Brand', sorted(df['GPU_company'].unique()))
        gpu_model = st.text_input('GPU Model (e.g., GTX 1050, Radeon RX 580)', 'Integrated')
        os = st.selectbox('Operating System', sorted(df['OS'].unique()))
    
    # Additional features
    with st.expander("Display Features"):
        touchscreen = st.checkbox('Touchscreen')
        ips_panel = st.checkbox('IPS Panel')
        retina_display = st.checkbox('Retina Display')
        resolution = st.selectbox('Screen Resolution', 
                                ['1366x768', '1920x1080', '2560x1600', '2880x1800', '3200x1800', '3840x2160'])
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Inches': [inches],
        'Ram': [ram],
        'Weight': [weight],
        'CPU_company': [cpu],
        'CPU_model': [cpu_model],
        'PrimaryStorage': [storage],
        'PrimaryStorageType': [storage_type],
        'GPU_company': [gpu],
        'GPU_model': [gpu_model],
        'OS': [os],
        'Touchscreen': [touchscreen],
        'IPSpanel': [ips_panel],
        'RetinaDisplay': [retina_display],
        'Screen': [resolution]
    })
    
    # EURO TO RUPEE CONVERSION RATE
    EURO_TO_RUPEE_RATE = 90.0  # Update this with current rate
    
    # Predict button with enhanced styling
    if st.button('Predict Price', type="primary", help="Click to predict the laptop price based on your specifications"):
        try:
            # Encode categorical variables for prediction
            input_data_encoded = input_data.copy()
            for col in ['Company', 'TypeName', 'CPU_company']:
                if col in label_encoders:
                    # Handle unseen labels by using the most common label
                    try:
                        input_data_encoded[col] = label_encoders[col].transform(input_data_encoded[col])
                    except ValueError:
                        # If label not seen during training, use the first available label
                        input_data_encoded[col] = 0
            
            # Select features for prediction
            features = ['Company', 'TypeName', 'Inches', 'Ram', 'Weight', 'CPU_company']
            prediction_input = input_data_encoded[features]
            
            # Make prediction
            prediction = model.predict(prediction_input)
            euro_price = prediction[0]
            rupee_price = euro_price * EURO_TO_RUPEE_RATE
            
            # Display results in two columns
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.success(f"### Predicted Price (Euros)\n€{euro_price:,.2f}")
            
            with col_result2:
                st.success(f"### Predicted Price (Rupees)\n₹{rupee_price:,.2f}")
            
            # Exchange rate information
            st.info(f"*Conversion rate: 1€ = ₹{EURO_TO_RUPEE_RATE} (approximate)*")
            
            # Show similar laptops from dataset
            st.subheader("Similar Laptops in Our Database")
            similar = df[
                (df['Company'] == company) & 
                (df['TypeName'] == type_name) & 
                (df['Ram'] == ram) & 
                (df['Inches'].between(inches-1, inches+1))
            ].sort_values('Price_euros')
            
            if not similar.empty:
                # Convert prices to rupees for display
                similar['Price_rupees'] = similar['Price_euros'] * EURO_TO_RUPEE_RATE
                st.dataframe(similar[['Company', 'Product', 'Ram', 'Inches', 'Price_euros', 'Price_rupees']].head(5))
            else:
                st.info("No similar laptops found in our database.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please check if all required features are available in the model.")

elif app_mode == "Data Exploration":
    st.title('📊 Laptop Data Exploration')
    
    st.subheader("Dataset Overview")
    st.write(f"Total laptops in dataset: {len(df)}")
    
    # Show raw data with filters
    st.subheader("Raw Data")
    with st.expander("Filter Data"):
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
            selected_types = st.multiselect(
                "Select Laptop Types",
                options=df['TypeName'].unique(),
                default=df['TypeName'].unique()[:2]
            )
    
    filtered_df = df[
        (df['Price_euros'].between(min_price, max_price)) &
        (df['Company'].isin(selected_brands)) &
        (df['Ram'].between(min_ram, max_ram)) &
        (df['TypeName'].isin(selected_types))
    ]
    
    st.write(f"Showing {len(filtered_df)} laptops")
    st.dataframe(filtered_df)
    
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
        sns.boxplot(
            x='Company', 
            y='Price_euros', 
            data=filtered_df,
            order=filtered_df.groupby('Company')['Price_euros'].median().sort_values().index,
            ax=ax
        )
        ax.set_title('Price Distribution by Brand')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    
    with tab3:
        corr_cols = ['Price_euros', 'Ram', 'Inches', 'Weight']
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            filtered_df[corr_cols].corr(),
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
    - **Data Exploration**: Explore the dataset with interactive filters and visualizations
    - **Similar Products**: Find similar laptops in our database
    
    ### How It Works:
    1. Select your desired laptop specifications
    2. Click "Predict Price" to get an estimated price
    3. Explore similar laptops in our database
    
    ### Data Source:
    The dataset contains information about various laptop models with their specifications and prices.
    
    ### Model Information:
    - Algorithm: Random Forest Regressor
    - Features used: Brand, Type, Screen Size, RAM, Weight, CPU, Storage, GPU
    """)
    
    st.markdown("---")
    st.markdown("Created with ❤️ using Python, Streamlit, and Scikit-learn")

