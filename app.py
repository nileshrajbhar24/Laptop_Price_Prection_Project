import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r"C:\Users\nitis\Analysis_Project\laptop_price_prediction\.ipynb_checkpoints\cleaned_laptop_prices.csv")

# Set page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Load data and models
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\nitis\Analysis_Project\laptop_price_prediction\.ipynb_checkpoints\cleaned_laptop_prices.csv')

@st.cache_resource
def load_model():
    model = pickle.load(open('laptop_model1.pkl', 'rb'))
    label_encoders = pickle.load(open('label_encoders1.pkl', 'rb'))
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
        cpu_model = st.text_input('CPU Model (e.g., Core i5, Ryzen 7)')
        storage = st.select_slider('Primary Storage (GB)', options=[32, 64, 128, 256, 512, 1024, 2048], value=256)
        storage_type = st.selectbox('Storage Type', ['SSD', 'HDD', 'Flash Storage', 'Hybrid'])
        gpu = st.selectbox('GPU Brand', sorted(df['GPU_company'].unique()))
        gpu_model = st.text_input('GPU Model (e.g., GTX 1050, Radeon RX 580)')
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
    
    # Encode categorical variables
    for col in ['Company', 'TypeName', 'CPU_company', 'GPU_company', 'OS', 'PrimaryStorageType', 'Screen']:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # EURO TO RUPEE CONVERSION RATE
    EURO_TO_RUPEE_RATE = 90.0  # Update this with current rate
    
    
    # Predict button with enhanced styling
        
    if st.button('Predict Price', type="primary", help="Click to predict the laptop price based on your specifications"):
        try:
            # Select only the features used in the model
            model_features = ['Company', 'TypeName', 'Inches', 'Ram', 'Weight', 
                            'CPU_company', 'PrimaryStorage', 'GPU_company']
            prediction_input = input_data[model_features]
            
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