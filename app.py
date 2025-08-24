import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/nileshrajbhar24/Laptop_Price_Prection_Project/refs/heads/main/cleaned_laptop_prices.csv"
    return pd.read_csv(url)

# Feature engineering with added enhancements
def engineer_features(df):
    df_eng = df.copy()
    df_eng['Screen_Size_Category'] = pd.cut(df_eng['Inches'], bins=[0, 13, 15, 17, 100],
                                           labels=['Small', 'Medium', 'Large', 'Extra Large'])
    df_eng['RAM_Category'] = pd.cut(df_eng['Ram'], bins=[0, 4, 8, 16, 100],
                                    labels=['Low', 'Medium', 'High', 'Very High'])
    df_eng['Storage_Category'] = pd.cut(df_eng['PrimaryStorage'], bins=[0, 256, 512, 1000, 10000],
                                        labels=['Low', 'Medium', 'High', 'Very High'])
    df_eng['Performance_Score'] = (df_eng['Ram'] / 8) + (df_eng['PrimaryStorage'] / 512) + (df_eng['Inches'] / 15)

    # Resolution parsing
    df_eng[['Resolution_Width', 'Resolution_Height']] = df_eng['ScreenResolution'].str.extract(r'(\d+)x(\d+)').astype(float)
    df_eng['TotalPixels'] = df_eng['Resolution_Width'] * df_eng['Resolution_Height']
    df_eng['Touchscreen'] = df_eng['ScreenResolution'].str.contains('Touchscreen', case=False).astype(int)
    df_eng['IPS'] = df_eng['ScreenResolution'].str.contains('IPS', case=False).astype(int)

    # Include CPU/GPU model names
    df_eng['CPU_Model'] = df_eng['Cpu']
    df_eng['GPU_Model'] = df_eng['Gpu']

    return df_eng

# Train model with engineered features
@st.cache_resource
def train_improved_model(df):
    df_eng = engineer_features(df)
    categorical_cols = [
        'Company', 'TypeName', 'CPU_company', 'GPU_company', 'OS',
        'Screen_Size_Category', 'RAM_Category', 'Storage_Category',
        'CPU_Model', 'GPU_Model'
    ]
    label_encoders = {}
    for col in categorical_cols:
        if col in df_eng.columns:
            le = LabelEncoder()
            df_eng[col] = le.fit_transform(df_eng[col].astype(str))
            label_encoders[col] = le

    features = [
        'Company', 'TypeName', 'Inches', 'Ram', 'Weight',
        'CPU_company', 'GPU_company', 'PrimaryStorage',
        'Screen_Size_Category', 'RAM_Category', 'Storage_Category',
        'Performance_Score', 'Touchscreen', 'IPS', 'TotalPixels',
        'CPU_Model', 'GPU_Model'
    ]
    features = [f for f in features if f in df_eng.columns]

    X = df_eng[features]
    y = df_eng['Price_euros']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae, r2 = mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)
    return model, label_encoders, {'mae': mae, 'r2': r2, 'features': features}

# Load and train
df = load_data()
model, label_encoders, model_metrics = train_improved_model(df)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a page", ["Price Prediction", "Data Exploration", "Model Info", "About"])

if app_mode == "Price Prediction":
    st.title('💻 Laptop Price Predictor')
    st.info(f"Model Performance: MAE = €{model_metrics['mae']:.2f}, R² = {model_metrics['r2']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Basic Specifications")
        company = st.selectbox('Brand', sorted(df['Company'].unique()))
        type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
        inches = st.slider('Screen Size (inches)', 10.0, 18.0, 15.6, 0.1)
        ram = st.select_slider('RAM (GB)', options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
        weight = st.slider('Weight (kg)', 0.5, 5.0, 2.0, 0.1)

    with col2:
        st.subheader("Advanced Specifications")
        cpu = st.selectbox('CPU Brand', sorted(df['CPU_company'].unique()))
        gpu = st.selectbox('GPU Brand', sorted(df['GPU_company'].unique()))
        storage = st.select_slider('Storage (GB)', options=[32, 64, 128, 256, 512, 1024, 2048], value=256)
        os = st.selectbox('Operating System', sorted(df['OS'].unique()))
        cpu_model = st.selectbox('CPU Model', sorted(df['Cpu'].unique()))
        gpu_model = st.selectbox('GPU Model', sorted(df['Gpu'].unique()))
        resolution = st.selectbox('Screen Resolution', sorted(df['ScreenResolution'].unique()))

    EURO_TO_RUPEE_RATE = 90.0

    if st.button('Predict Price'):
        # Derived features for input
        screen_size_cat = pd.cut([inches], bins=[0,13,15,17,100],
                                 labels=['Small','Medium','Large','Extra Large'])[0]
        ram_cat = pd.cut([ram], bins=[0,4,8,16,100],
                         labels=['Low','Medium','High','Very High'])[0]
        storage_cat = pd.cut([storage], bins=[0,256,512,1000,10000],
                             labels=['Low','Medium','High','Very High'])[0]
        performance_score = (ram / 8) + (storage / 512) + (inches / 15)

        try:
            w, h = map(int, resolution.split('x'))
            total_pixels = w * h
        except:
            w, h, total_pixels = 1920, 1080, 1920 * 1080

        touchscreen = int("touchscreen" in resolution.lower())
        ips = int("ips" in resolution.lower())

        input_data = pd.DataFrame({
            'Company': [company], 'TypeName': [type_name], 'Inches': [inches], 'Ram': [ram], 'Weight': [weight],
            'CPU_company': [cpu], 'GPU_company': [gpu], 'PrimaryStorage': [storage],
            'Screen_Size_Category': [screen_size_cat], 'RAM_Category': [ram_cat], 'Storage_Category': [storage_cat],
            'Performance_Score': [performance_score], 'Touchscreen': [touchscreen], 'IPS': [ips],
            'TotalPixels': [total_pixels], 'CPU_Model': [cpu_model], 'GPU_Model': [gpu_model]
        })

        input_encoded = input_data.copy()
        for col, le in label_encoders.items():
            if col in input_encoded.columns:
                input_encoded[col] = input_encoded[col].map(lambda v: le.transform([v])[0] if v in le.classes_ else 0)

        for feat in model_metrics['features']:
            if feat not in input_encoded:
                input_encoded[feat] = 0

        X_input = input_encoded[model_metrics['features']]
        euro_price = model.predict(X_input)[0]
        rupee_price = euro_price * EURO_TO_RUPEE_RATE

        c1, c2 = st.columns(2)
        with c1:
            st.success(f"### Predicted Price (Euros)\n€{euro_price:,.2f}")
            st.metric("Mean Absolute Error", f"€{model_metrics['mae']:.2f}")
        with c2:
            st.success(f"### Predicted Price (Rupees)\n₹{rupee_price:,.2f}")
            st.metric("R² Score", f"{model_metrics['r2']:.3f}")
        st.info(f"*Conversion rate: 1€ = ₹{EURO_TO_RUPEE_RATE}*")

        st.subheader("Similar Laptops in Database")
        similar = df[(df['Company']==company) & (df['TypeName']==type_name) &
                     (df['Ram']==ram) & (df['PrimaryStorage']==storage)].head(5)
        if similar.empty:
            similar = df[(df['Company']==company) & (df['TypeName']==type_name)].head(5)
            st.info("Showing similar laptops from the same brand and type:" if not similar.empty else "No similar laptops found in our database.")
        if not similar.empty:
            similar['Price_rupees'] = similar['Price_euros'] * EURO_TO_RUPEE_RATE
            st.dataframe(similar[['Company','TypeName','Ram','Inches','PrimaryStorage','Price_euros','Price_rupees']])

elif app_mode == "Data Exploration":
    st.title('📊 Laptop Data Exploration')
    st.write(f"Dataset contains {len(df)} laptops")
    col1, col2 = st.columns(2)
    with col1:
        min_price, max_price = st.slider("Price Range (€)", int(df['Price_euros'].min()), int(df['Price_euros'].max()), (int(df['Price_euros'].min()), int(df['Price_euros'].max())))
        selected_brands = st.multiselect("Select Brands", options=df['Company'].unique(), default=list(df['Company'].unique())[:3])
    with col2:
        min_ram, max_ram = st.slider("RAM Range (GB)", int(df['Ram'].min()), int(df['Ram'].max()), (int(df['Ram'].min()), int(df['Ram'].max())))
        selected_types = st.multiselect("Select Types", options=df['TypeName'].unique(), default=list(df['TypeName'].unique())[:2])

    filtered = df[(df['Price_euros'].between(min_price, max_price)) &
                  (df['Company'].isin(selected_brands)) &
                  (df['Ram'].between(min_ram, max_ram)) &
                  (df['TypeName'].isin(selected_types))]

    st.write(f"Showing {len(filtered)} laptops")
    st.dataframe(filtered.head(20))

    st.subheader("Visualizations")
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Brand Comparison", "Feature Correlations"])
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered['Price_euros'], bins=20, kde=True, ax=ax)
        ax.set(title='Price Distribution', xlabel='Price (€)', ylabel='Count')
        st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 6))
        avg = filtered.groupby('Company')['Price_euros'].mean().sort_values(ascending=False)
        sns.barplot(x=avg.index, y=avg.values, ax=ax)
        ax.set(title='Average Price by Brand')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    with tab3:
        num_cols = ['Price_euros','Ram','Inches','Weight','PrimaryStorage']
        corr = filtered[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlations')
        st.pyplot(fig)

elif app_mode == "Model Info":
    st.title('🤖 Model Information')
    st.metric("Mean Absolute Error", f"€{model_metrics['mae']:.2f}")
    st.metric("R² Score", f"{model_metrics['r2']:.3f}")
    st.subheader("Features Used:")
    for i, feat in enumerate(model_metrics['features'], 1):
        st.write(f"{i}. {feat}")
    st.subheader("How to Improve:")
    st.markdown("""
    1. More data  
    2. Granular specifications (detailed CPU/GPU, PPI)  
    3. Hyperparameter tuning  
    4. More feature engineering  
    5. Try other algorithms (Gradient Boosting, Neural Nets)
    """)

elif app_mode == "About":
    st.title('ℹ️ About This Project')
    st.markdown("""
    This app predicts laptop prices using Streamlit and an enhanced RandomForest model.
    **Features**:
    - Price prediction with extended spec inputs  
    - Quick data exploration  
    - Details about the model and improvement ideas  
    - Finds similar laptops in the dataset  

    **Technologies**:
    - Python, Pandas, scikit-learn, Streamlit  
    - Feature engineering includes resolution, touchscreen, IPS, CPU/GPU models  
    """)
    st.markdown("---")
    st.markdown("Built with ❤️ using Python and Streamlit")
