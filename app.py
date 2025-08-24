import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Load data from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/nileshrajbhar24/Laptop_Price_Prection_Project/refs/heads/main/cleaned_laptop_prices.csv"
    return pd.read_csv(url)

# Enhanced feature engineering
def engineer_features(df):
    df_eng = df.copy()
    
    # Create screen size categories
    df_eng['Screen_Size_Category'] = pd.cut(df_eng['Inches'], 
                                           bins=[0, 13, 15, 17, 100], 
                                           labels=['Small', 'Medium', 'Large', 'Extra Large'])
    
    # Create RAM categories
    df_eng['RAM_Category'] = pd.cut(df_eng['Ram'], 
                                   bins=[0, 4, 8, 16, 100], 
                                   labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Create storage categories
    df_eng['Storage_Category'] = pd.cut(df_eng['PrimaryStorage'], 
                                       bins=[0, 256, 512, 1000, 10000], 
                                       labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Create performance score (simple heuristic)
    df_eng['Performance_Score'] = (df_eng['Ram'] / 8) + (df_eng['PrimaryStorage'] / 512) + (df_eng['Inches'] / 15)
    
    return df_eng

# Train an improved model with better features
@st.cache_resource
def train_improved_model(df):
    # Engineer features
    df_eng = engineer_features(df)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Company', 'TypeName', 'CPU_company', 'GPU_company', 'OS', 
                       'Screen_Size_Category', 'RAM_Category', 'Storage_Category']
    
    for col in categorical_cols:
        if col in df_eng.columns:
            le = LabelEncoder()
            df_eng[col] = le.fit_transform(df_eng[col].astype(str))
            label_encoders[col] = le
    
    # Select features and target
    features = [
        'Company', 'TypeName', 'Inches', 'Ram', 'Weight', 'CPU_company', 
        'GPU_company', 'PrimaryStorage', 'Screen_Size_Category', 
        'RAM_Category', 'Storage_Category', 'Performance_Score'
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
        max_depth=15, 
        min_samples_split=5, 
        min_samples_leaf=2, 
        random_state=42
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
        'features': features
    }
    
    return model, label_encoders, model_metrics

# Load data and train model
df = load_data()
model, label_encoders, model_metrics = train_improved_model(df)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a page", ["Price Prediction", "Data Exploration", "Model Info", "About"])

if app_mode == "Price Prediction":
    st.title('💻 Laptop Price Predictor')
    
    # Display model performance
    st.info(f"Model Performance: MAE = €{model_metrics['mae']:.2f}, R² = {model_metrics['r2']:.3f}")
    
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
            # Calculate derived features
            screen_size_cat = pd.cut([inches], bins=[0, 13, 15, 17, 100], 
                                    labels=['Small', 'Medium', 'Large', 'Extra Large'])[0]
            ram_cat = pd.cut([ram], bins=[0, 4, 8, 16, 100], 
                            labels=['Low', 'Medium', 'High', 'Very High'])[0]
            storage_cat = pd.cut([storage], bins=[0, 256, 512, 1000, 10000], 
                                labels=['Low', 'Medium', 'High', 'Very High'])[0]
            performance_score = (ram / 8) + (storage / 512) + (inches / 15)
            
            # Prepare input data with ALL required features
            input_data = pd.DataFrame({
                'Company': [company],
                'TypeName': [type_name],
                'Inches': [inches],
                'Ram': [ram],
                'Weight': [weight],
                'CPU_company': [cpu],
                'GPU_company': [gpu],
                'PrimaryStorage': [storage],
                'Screen_Size_Category': [screen_size_cat],
                'RAM_Category': [ram_cat],
                'Storage_Category': [storage_cat],
                'Performance_Score': [performance_score]
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
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"### Predicted Price (Euros)\n€{euro_price:,.2f}")
                st.metric("Mean Absolute Error", f"€{model_metrics['mae']:.2f}")
            with col2:
                st.success(f"### Predicted Price (Rupees)\n₹{rupee_price:,.2f}")
                st.metric("R² Score", f"{model_metrics['r2']:.3f}")
            
            st.info(f"*Conversion rate: 1€ = ₹{EURO_TO_RUPEE_RATE}*")
            
            # Show similar laptops
            st.subheader("Similar Laptops in Database")
            similar = df[
                (df['Company'] == company) & 
                (df['TypeName'] == type_name) & 
                (df['Ram'] == ram) &
                (df['PrimaryStorage'] == storage)
            ].head(5)
            
            if not similar.empty:
                similar['Price_rupees'] = similar['Price_euros'] * EURO_TO_RUPEE_RATE
                st.dataframe(similar[['Company', 'TypeName', 'Ram', 'Inches', 'PrimaryStorage', 'Price_euros', 'Price_rupees']])
            else:
                # Show broader similarity if exact match not found
                similar = df[
                    (df['Company'] == company) & 
                    (df['TypeName'] == type_name)
                ].head(5)
                if not similar.empty:
                    st.info("Showing similar laptops from the same brand and type:")
                    similar['Price_rupees'] = similar['Price_euros'] * EURO_TO_RUPEE_RATE
                    st.dataframe(similar[['Company', 'TypeName', 'Ram', 'Inches', 'PrimaryStorage', 'Price_euros', 'Price_rupees']])
                else:
                    st.info("No similar laptops found in our database.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try adjusting your specifications.")

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
        selected_types = st.multiselect(
            "Select Types",
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
        ax.set_ylabel('Average Price (€)')
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
            center=0,
            ax=ax
        )
        ax.set_title('Feature Correlations')
        st.pyplot(fig)

elif app_mode == "Model Info":
    st.title('🤖 Model Information')
    
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Absolute Error", f"€{model_metrics['mae']:.2f}")
    with col2:
        st.metric("R² Score", f"{model_metrics['r2']:.3f}")
    
    st.subheader("Features Used for Prediction")
    st.write("The model uses the following features to predict laptop prices:")
    for i, feature in enumerate(model_metrics['features'], 1):
        st.write(f"{i}. {feature}")
    
    st.subheader("How to Improve Accuracy")
    st.markdown("""
    1. **More Data**: Add more laptop examples to the dataset
    2. **Better Features**: Include more detailed specifications like CPU model, GPU model, etc.
    3. **Hyperparameter Tuning**: Optimize model parameters further
    4. **Feature Engineering**: Create more informative derived features
    5. **Different Algorithms**: Try other regression algorithms like Gradient Boosting or Neural Networks
    """)

elif app_mode == "About":
    st.title('ℹ️ About This Project')
    
    st.markdown("""
    ## Laptop Price Predictor
    
    This application predicts laptop prices based on their specifications using machine learning.
    
    ### Features:
    - **Price Prediction**: Estimate laptop prices based on specifications
    - **Data Exploration**: Explore the dataset with filters and visualizations
    - **Model Information**: View model performance and details
    - **Similar Products**: Find similar laptops in our database
    
    ### How It Works:
    1. Select your desired laptop specifications
    2. Click "Predict Price" to get an estimated price
    3. Explore similar laptops in our database
    
    ### Data Source:
    The dataset contains information about various laptop models with their specifications and prices.
    
    ### Model Information:
    - Algorithm: Random Forest Regressor
    - Features used: Brand, Type, Screen Size, RAM, Weight, CPU, GPU, Storage, and engineered features
    """)
    
    st.markdown("---")
    st.markdown("Created with ❤️ using Python and Streamlit")
