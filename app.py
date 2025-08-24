import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Advanced Laptop Price Predictor", page_icon="💻", layout="wide")

# Load data from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/nileshrajbhar24/Laptop_Price_Prection_Project/refs/heads/main/cleaned_laptop_prices.csv"
    return pd.read_csv(url)

# Enhanced feature engineering with detailed features
def engineer_detailed_features(df):
    df_eng = df.copy()
    
    # Extract CPU details
    def extract_cpu_details(cpu_name):
        if pd.isna(cpu_name):
            return 'Unknown', 'Unknown', 0
        
        cpu_name = str(cpu_name).lower()
        
        # Extract generation
        generation = 0
        gen_match = re.search(r'(\d+)th gen', cpu_name)
        if gen_match:
            generation = int(gen_match.group(1))
        else:
            gen_match = re.search(r'i[-\s]?(\d+)', cpu_name)
            if gen_match:
                generation = int(gen_match.group(1))
        
        # Extract CPU tier (i3, i5, i7, i9, Ryzen 3, 5, 7, 9)
        tier = 'Other'
        if 'i3' in cpu_name or 'core i3' in cpu_name:
            tier = 'i3'
        elif 'i5' in cpu_name or 'core i5' in cpu_name:
            tier = 'i5'
        elif 'i7' in cpu_name or 'core i7' in cpu_name:
            tier = 'i7'
        elif 'i9' in cpu_name or 'core i9' in cpu_name:
            tier = 'i9'
        elif 'ryzen 3' in cpu_name:
            tier = 'Ryzen 3'
        elif 'ryzen 5' in cpu_name:
            tier = 'Ryzen 5'
        elif 'ryzen 7' in cpu_name:
            tier = 'Ryzen 7'
        elif 'ryzen 9' in cpu_name:
            tier = 'Ryzen 9'
        
        # Extract performance level
        performance = 'Standard'
        if 'u' in cpu_name or 'ultra' in cpu_name:
            performance = 'Low Power'
        elif 'h' in cpu_name or 'hq' in cpu_name or 'hk' in cpu_name:
            performance = 'High Performance'
        elif 'g' in cpu_name:
            performance = 'Graphics Focused'
        
        return tier, performance, generation
    
    # Apply CPU feature extraction
    cpu_details = df_eng['CPU_model'].apply(extract_cpu_details)
    df_eng['CPU_Tier'] = [detail[0] for detail in cpu_details]
    df_eng['CPU_Performance'] = [detail[1] for detail in cpu_details]
    df_eng['CPU_Generation'] = [detail[2] for detail in cpu_details]
    
    # Extract GPU details
    def extract_gpu_details(gpu_name):
        if pd.isna(gpu_name):
            return 'Integrated', 'Entry'
        
        gpu_name = str(gpu_name).lower()
        
        # GPU type
        gpu_type = 'Dedicated'
        if 'integrated' in gpu_name or 'intel' in gpu_name or 'uhd' in gpu_name or 'iris' in gpu_name:
            gpu_type = 'Integrated'
        elif 'radeon' in gpu_name or 'rtx' in gpu_name or 'gtx' in gpu_name:
            gpu_type = 'Dedicated'
        
        # GPU performance level
        performance = 'Entry'
        if 'rtx' in gpu_name or 'gtx 16' in gpu_name or 'rx' in gpu_name:
            performance = 'Mid-Range'
        if 'rtx 30' in gpu_name or 'rtx 40' in gpu_name or 'rx 6' in gpu_name or 'rx 7' in gpu_name:
            performance = 'High-End'
        if 'quadro' in gpu_name or 'rtx a' in gpu_name:
            performance = 'Workstation'
        
        return gpu_type, performance
    
    # Apply GPU feature extraction
    gpu_details = df_eng['GPU_model'].apply(extract_gpu_details)
    df_eng['GPU_Type'] = [detail[0] for detail in gpu_details]
    df_eng['GPU_Performance'] = [detail[1] for detail in gpu_details]
    
    # Process screen resolution
    def parse_resolution(res_str):
        if pd.isna(res_str):
            return 1920, 1080
        
        try:
            if 'x' in str(res_str):
                width, height = map(int, str(res_str).split('x'))
                return width, height
            else:
                return 1920, 1080
        except:
            return 1920, 1080
    
    # Extract resolution details
    resolution_details = df_eng['Screen'].apply(parse_resolution)
    df_eng['Screen_Width'] = [res[0] for res in resolution_details]
    df_eng['Screen_Height'] = [res[1] for res in resolution_details]
    df_eng['Screen_Pixels'] = df_eng['Screen_Width'] * df_eng['Screen_Height']
    df_eng['PPI'] = df_eng['Screen_Pixels'] / (df_eng['Inches'] ** 2)
    
    # Create performance score
    df_eng['Performance_Score'] = (
        (df_eng['Ram'] / 16) +  # Normalize RAM (16GB = 1.0)
        (df_eng['PrimaryStorage'] / 1000) +  # Normalize Storage (1TB = 1.0)
        (df_eng['CPU_Generation'] / 10) +  # Normalize CPU generation
        (df_eng['Inches'] / 17)  # Normalize screen size
    )
    
    # Premium brand flag
    premium_brands = ['Apple', 'Dell XPS', 'Razer', 'Microsoft', 'Lenovo ThinkPad']
    df_eng['Is_Premium_Brand'] = df_eng['Company'].isin(premium_brands)
    
    return df_eng

# Train an improved model with detailed features
@st.cache_resource
def train_detailed_model(df):
    # Engineer detailed features
    df_eng = engineer_detailed_features(df)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = [
        'Company', 'TypeName', 'CPU_company', 'GPU_company', 'OS',
        'CPU_Tier', 'CPU_Performance', 'GPU_Type', 'GPU_Performance'
    ]
    
    for col in categorical_cols:
        if col in df_eng.columns:
            le = LabelEncoder()
            df_eng[col] = le.fit_transform(df_eng[col].astype(str))
            label_encoders[col] = le
    
    # Select features and target
    features = [
        'Company', 'TypeName', 'Inches', 'Ram', 'Weight', 'CPU_company', 
        'GPU_company', 'PrimaryStorage', 'CPU_Tier', 'CPU_Generation',
        'GPU_Type', 'GPU_Performance', 'Screen_Width', 'Screen_Height',
        'Screen_Pixels', 'PPI', 'Performance_Score', 'Is_Premium_Brand'
    ]
    
    # Only use features that exist in the dataframe
    features = [f for f in features if f in df_eng.columns]
    
    X = df_eng[features]
    y = df_eng['Price_euros']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=20, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store evaluation metrics
    model_metrics = {
        'mae': mae,
        'r2': r2,
        'features': features,
        'feature_importance': dict(zip(features, model.feature_importances_))
    }
    
    return model, label_encoders, model_metrics

# Load data and train model
df = load_data()
model, label_encoders, model_metrics = train_detailed_model(df)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a page", ["Price Prediction", "Data Exploration", "Model Info", "About"])

if app_mode == "Price Prediction":
    st.title('💻 Advanced Laptop Price Predictor')
    
    # Display model performance
    st.info(f"Model Performance: MAE = €{model_metrics['mae']:.2f}, R² = {model_metrics['r2']:.3f}")
    
    # Create tabs for different specification categories
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Specs", "CPU/GPU Details", "Display Features", "Advanced Options"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Specifications")
            company = st.selectbox('Brand', sorted(df['Company'].unique()))
            type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
            inches = st.slider('Screen Size (inches)', min_value=10.0, max_value=18.0, value=15.6, step=0.1)
        with col2:
            ram = st.select_slider('RAM (GB)', options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
            weight = st.slider('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
            storage = st.select_slider('Storage (GB)', options=[32, 64, 128, 256, 512, 1024, 2048], value=256)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("CPU Details")
            cpu_brand = st.selectbox('CPU Brand', sorted(df['CPU_company'].unique()))
            cpu_tier = st.selectbox('CPU Tier', ['i3', 'i5', 'i7', 'i9', 'Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9', 'Other'])
            cpu_gen = st.slider('CPU Generation', min_value=1, max_value=15, value=11)
            cpu_perf = st.selectbox('CPU Performance', ['Low Power', 'Standard', 'High Performance'])
        with col2:
            st.subheader("GPU Details")
            gpu_brand = st.selectbox('GPU Brand', sorted(df['GPU_company'].unique()))
            gpu_type = st.selectbox('GPU Type', ['Integrated', 'Dedicated'])
            gpu_perf = st.selectbox('GPU Performance', ['Entry', 'Mid-Range', 'High-End', 'Workstation'])
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Display Features")
            resolution = st.selectbox('Screen Resolution', 
                                    ['1366x768', '1920x1080', '2560x1440', '2560x1600', 
                                     '2880x1800', '3200x1800', '3840x2160', '3840x2400'])
            touchscreen = st.checkbox('Touchscreen')
            ips_panel = st.checkbox('IPS Panel')
        with col2:
            retina = st.checkbox('Retina Display')
            refresh_rate = st.selectbox('Refresh Rate', ['60Hz', '120Hz', '144Hz', '240Hz', '360Hz'])
            hdr = st.checkbox('HDR Support')
    
    with tab4:
        st.subheader("Additional Features")
        os = st.selectbox('Operating System', sorted(df['OS'].unique()))
        premium_brand = st.checkbox('Premium Brand (Apple, Dell XPS, Razer, etc.)')
    
    # EURO TO RUPEE CONVERSION RATE
    EURO_TO_RUPEE_RATE = 90.0
    
    if st.button('Predict Price', type="primary"):
        try:
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            pixels = width * height
            ppi = pixels / (inches ** 2)
            
            # Calculate performance score
            performance_score = (
                (ram / 16) +  # Normalize RAM (16GB = 1.0)
                (storage / 1000) +  # Normalize Storage (1TB = 1.0)
                (cpu_gen / 10) +  # Normalize CPU generation
                (inches / 17)  # Normalize screen size
            )
            
            # Prepare input data with ALL required features
            input_data = pd.DataFrame({
                'Company': [company],
                'TypeName': [type_name],
                'Inches': [inches],
                'Ram': [ram],
                'Weight': [weight],
                'CPU_company': [cpu_brand],
                'GPU_company': [gpu_brand],
                'PrimaryStorage': [storage],
                'CPU_Tier': [cpu_tier],
                'CPU_Generation': [cpu_gen],
                'CPU_Performance': [cpu_perf],
                'GPU_Type': [gpu_type],
                'GPU_Performance': [gpu_perf],
                'Screen_Width': [width],
                'Screen_Height': [height],
                'Screen_Pixels': [pixels],
                'PPI': [ppi],
                'Performance_Score': [performance_score],
                'Is_Premium_Brand': [premium_brand]
            })
            
            # Encode categorical variables
            input_data_encoded = input_data.copy()
            for col in label_encoders.keys():
                if col in input_data_encoded.columns:
                    try:
                        input_data_encoded[col] = label_encoders[col].transform(input_data_encoded[col])
                    except ValueError:
                        # If label not seen during training, use the first available label
                        input_data_encoded[col] = 0
            
            # Make sure we have all the features the model expects
            for feature in model_metrics['features']:
                if feature not in input_data_encoded.columns:
                    input_data_encoded[feature] = 0
            
            # Select only the features the model was trained on
            prediction_input = input_data_encoded[model_metrics['features']]
            
            # Make prediction
            prediction = model.predict(prediction_input)
            euro_price = prediction[0]
            rupee_price = euro_price * EURO_TO_RUPEE_RATE
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"### Predicted Price (Euros)\n€{euro_price:,.2f}")
            with col2:
                st.success(f"### Predicted Price (Rupees)\n₹{rupee_price:,.2f}")
            with col3:
                st.metric("Mean Absolute Error", f"€{model_metrics['mae']:.2f}")
                st.metric("R² Score", f"{model_metrics['r2']:.3f}")
            
            st.info(f"*Conversion rate: 1€ = ₹{EURO_TO_RUPEE_RATE}*")
            
            # Show feature importance
            with st.expander("What influenced this price the most?"):
                importance_df = pd.DataFrame({
                    'Feature': model_metrics['feature_importance'].keys(),
                    'Importance': model_metrics['feature_importance'].values()
                }).sort_values('Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                ax.set_title('Top 10 Features Influencing Price')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try adjusting your specifications.")

elif app_mode == "Data Exploration":
    st.title('📊 Laptop Data Exploration')
    
    # Enhanced data exploration with detailed features
    df_eng = engineer_detailed_features(df)
    
    st.subheader("Dataset Overview")
    st.write(f"Total laptops in dataset: {len(df_eng)}")
    
    # Show feature distributions
    st.subheader("Feature Distributions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Price", f"€{df_eng['Price_euros'].mean():.2f}")
    with col2:
        st.metric("Average RAM", f"{df_eng['Ram'].mean():.1f} GB")
    with col3:
        st.metric("Average Storage", f"{df_eng['PrimaryStorage'].mean():.0f} GB")
    
    # Interactive visualizations
    st.subheader("Interactive Analysis")
    
    viz_option = st.selectbox("Choose visualization", 
                             ["Price by CPU Tier", "Price by GPU Type", 
                              "Price vs Performance Score", "Brand Comparison"])
    
    if viz_option == "Price by CPU Tier":
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='CPU_Tier', y='Price_euros', data=df_eng, ax=ax)
        ax.set_title('Price Distribution by CPU Tier')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
    elif viz_option == "Price by GPU Type":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='GPU_Type', y='Price_euros', data=df_eng, ax=ax)
        ax.set_title('Price Distribution by GPU Type')
        st.pyplot(fig)
        
    elif viz_option == "Price vs Performance Score":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Performance_Score', y='Price_euros', data=df_eng, alpha=0.6, ax=ax)
        ax.set_title('Price vs Performance Score')
        st.pyplot(fig)
        
    elif viz_option == "Brand Comparison":
        fig, ax = plt.subplots(figsize=(14, 6))
        brand_avg = df_eng.groupby('Company')['Price_euros'].mean().sort_values(ascending=False)
        sns.barplot(x=brand_avg.index, y=brand_avg.values, ax=ax)
        ax.set_title('Average Price by Brand')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

elif app_mode == "Model Info":
    st.title('🤖 Model Information')
    
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Absolute Error", f"€{model_metrics['mae']:.2f}")
        st.write("Average prediction error")
    with col2:
        st.metric("R² Score", f"{model_metrics['r2']:.3f}")
        st.write("Variance explained by model")
    
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': model_metrics['feature_importance'].keys(),
        'Importance': model_metrics['feature_importance'].values()
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax)
    ax.set_title('Top 15 Features by Importance')
    st.pyplot(fig)
    
    st.subheader("Features Used")
    st.write("The model uses these detailed features for prediction:")
    for i, (feature, importance) in enumerate(importance_df.head(20).iterrows(), 1):
        st.write(f"{i}. {feature['Feature']} (Importance: {feature['Importance']:.4f})")

elif app_mode == "About":
    st.title('ℹ️ About This Advanced Predictor')
    
    st.markdown("""
    ## Advanced Laptop Price Predictor
    
    This application uses detailed feature engineering to provide more accurate laptop price predictions.
    
    ### Enhanced Features:
    - **CPU Details**: Tier (i3/i5/i7/i9), generation, performance level
    - **GPU Details**: Type (Integrated/Dedicated), performance level
    - **Display Features**: Resolution, PPI, pixel count
    - **Performance Scoring**: Combined metric based on specifications
    
    ### Technical Details:
    - Algorithm: Random Forest Regressor with 200 trees
    - Feature Engineering: Advanced parsing of CPU/GPU models
    - Performance: Lower MAE and higher R² through detailed features
    """)
    
    st.markdown("---")
    st.markdown("Created with ❤️ using Python, Streamlit, and Scikit-learn")

# Add some custom CSS for better styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stSuccess {
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)
