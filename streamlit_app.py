import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For model saving/loading
import shap # For SHAP plots
from sklearn.base import clone # For cloning estimators
from sklearn.ensemble import RandomForestClassifier  # SADECE RandomForest buradan
from xgboost import XGBClassifier                    # XGBClassifier buradan
from sklearn.tree import DecisionTreeClassifier      # DecisionTreeClassifier buradan


# Assuming titanic_model.py is in the same directory
# Ensure titanic_model.py can be imported (e.g., it doesn't run main() on import unconditionally)
try:
    from titanic_model import TitanicPredictor
except ImportError as e:
    st.error(f"Could not import TitanicPredictor from titanic_model.py: {e}")
    st.error("Please ensure titanic_model.py is in the same directory and is importable.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

# --- Global Matplotlib/Seaborn Settings (from titanic_model.py) ---
# Seaborn styles might need version adjustment e.g. 'seaborn-v0_8-darkgrid'
try:
    plt.style.use('seaborn-v0_8-darkgrid') 
except:
    try:
        plt.style.use('seaborn')
    except:
        st.warning("Seaborn style 'seaborn-v0_8-darkgrid' or 'seaborn' not found. Using default.")
sns.set_palette("husl")

# --- Helper Functions ---
@st.cache_resource # Cache the predictor object for the session
def get_predictor_instance():
    return TitanicPredictor()

def initialize_session_state():
    if 'predictor_initialized' not in st.session_state:
        st.session_state.predictor = get_predictor_instance()
        st.session_state.predictor_initialized = True
    
    # Data states
    if 'train_df' not in st.session_state: st.session_state.train_df = None
    if 'test_df' not in st.session_state: st.session_state.test_df = None
    if 'y_train_full' not in st.session_state: st.session_state.y_train_full = None
    if 'test_passenger_ids' not in st.session_state: st.session_state.test_passenger_ids = None
    
    # Processed data states
    if 'processed_train_df' not in st.session_state: st.session_state.processed_train_df = None
    if 'processed_test_df' not in st.session_state: st.session_state.processed_test_df = None
    if 'X_train_scaled_df' not in st.session_state: st.session_state.X_train_scaled_df = None
    if 'X_test_scaled' not in st.session_state: st.session_state.X_test_scaled = None # This will be a NumPy array
    if 'features_for_scaling' not in st.session_state: st.session_state.features_for_scaling = None
        
    # Model/Training states
    if 'model_trained' not in st.session_state: st.session_state.model_trained = False
    if 'best_model_name' not in st.session_state: st.session_state.best_model_name = None
    if 'model_performance_results' not in st.session_state: st.session_state.model_performance_results = []

initialize_session_state()
predictor = st.session_state.predictor


def reset_app_state():
    # Clear all relevant session state variables to start fresh
    keys_to_reset = [
        'predictor_initialized', 'train_df', 'test_df', 'y_train_full', 
        'test_passenger_ids', 'processed_train_df', 'processed_test_df',
        'X_train_scaled_df', 'X_test_scaled', 'features_for_scaling',
        'model_trained', 'best_model_name', 'model_performance_results'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # Re-initialize predictor
    st.cache_resource.clear() # Clear resource cache for predictor
    initialize_session_state() # Re-initialize
    st.rerun()


# --- Main Application ---
st.title("🚢 Titanic Survival Prediction Interactive Dashboard")

# --- Sidebar for Navigation/Controls ---
st.sidebar.title("🚢 Titanic Survival Prediction")
st.sidebar.markdown("""
Bu uygulama, Titanic yolcularının hayatta kalıp kalamayacağını makine öğrenmesi ile tahmin eder. 
Veri analizi, model karşılaştırma ve bireysel tahmin özellikleri sunar.
""")

if st.sidebar.button("🔄 Reset All & Start Over"):
    reset_app_state()

st.sidebar.header("1. Data Loading")
default_train_file = 'train (2).csv'
default_test_file = 'test (2).csv'

# Radio button for data source selection
data_source = st.sidebar.radio(
    "Veri kaynağı seç:",
    ('Varsayılan dosyaları kullan', 'CSV dosyası yükle'),
    key='data_source_selection'
)

train_file_path = default_train_file
test_file_path = default_test_file

if data_source == 'CSV dosyası yükle':
    uploaded_train_file = st.sidebar.file_uploader("Eğitim CSV", type="csv", key="train_uploader")
    uploaded_test_file = st.sidebar.file_uploader("Test CSV", type="csv", key="test_uploader")
    if uploaded_train_file: train_file_path = uploaded_train_file
    if uploaded_test_file: test_file_path = uploaded_test_file
else: # Use default
    st.sidebar.markdown(f"Varsayılan eğitim: `{default_train_file}`")
    st.sidebar.markdown(f"Varsayılan test: `{default_test_file}`")


# --- Dinamik filtreler ---
st.sidebar.header("Veri Filtresi")
filter_sex = st.sidebar.multiselect("Cinsiyet", ['male', 'female'], default=['male', 'female'])
filter_pclass = st.sidebar.multiselect("Sınıf", [1,2,3], default=[1,2,3])
filter_age = st.sidebar.slider("Yaş Aralığı", 0, 80, (0,80))

# --- Model seçimi ---
st.sidebar.header("Model Seçimi")
model_options = ["Otomatik (En iyi CV F1)", "Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]
selected_model = st.sidebar.selectbox("Kullanılacak Model", model_options, index=0)

# --- Özet kutuları için yardımcı fonksiyon ---
def summary_card(title, value, color='#4F8BF9'):
    st.markdown(f"""
    <div style='background-color:{color};padding:10px 20px;border-radius:10px;margin-bottom:10px'>
        <h4 style='color:white;margin-bottom:0'>{title}</h4>
        <h2 style='color:white;margin-top:0'>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

if st.sidebar.button("Load Data", key="load_data_button"):
    # Reset relevant parts of state before loading new data, but keep predictor object
    for key in ['train_df', 'test_df', 'y_train_full', 'test_passenger_ids', 
                'processed_train_df', 'processed_test_df', 'X_train_scaled_df', 
                'X_test_scaled', 'features_for_scaling', 'model_trained', 
                'best_model_name', 'model_performance_results']:
        if key in st.session_state:
            st.session_state[key] = None if key.endswith('_df') or key == 'features_for_scaling' or key.startswith('X_') else (False if key == 'model_trained' else ([] if key.endswith('_results') else None))
    
    # Re-initialize the predictor's internal state if needed (e.g. label encoders) by creating a new one
    st.session_state.predictor = get_predictor_instance() # Get a fresh predictor state
    predictor = st.session_state.predictor


    train_df_loaded = predictor.load_data(train_file_path)
    test_df_loaded = predictor.load_data(test_file_path)

    if train_df_loaded is not None:
        st.session_state.train_df = train_df_loaded
        if 'Survived' in train_df_loaded.columns:
            st.session_state.y_train_full = train_df_loaded['Survived'].copy()
            st.sidebar.success("Training data loaded.")
            # For EDA, predictor.data needs to be set (original script behavior)
            predictor.data = st.session_state.train_df.copy() 
        else:
            st.sidebar.error("'Survived' column not found in training data.")
            st.session_state.train_df = None # Invalidate
    else:
        st.sidebar.error("Failed to load training data.")

    if test_df_loaded is not None:
        st.session_state.test_df = test_df_loaded
        if 'PassengerId' in test_df_loaded.columns:
            st.session_state.test_passenger_ids = test_df_loaded['PassengerId'].copy()
        else:
            st.sidebar.warning("PassengerId not found in test data. Submission file might be affected. Generating dummy IDs.")
            st.session_state.test_passenger_ids = pd.Series(range(1, len(test_df_loaded) + 1), name="PassengerId")
        st.sidebar.success("Test data loaded.")
    else:
        st.sidebar.error("Failed to load test data.")
    st.rerun()


# --- EDA (Keşifsel Veri Analizi) Gelişmiş ---
def plot_eda_graphs(df):
    st.subheader("📊 Gelişmiş EDA (Keşifsel Veri Analizi) Grafikleri")
    st.write("Aşağıdan görmek istediğiniz grafikleri seçebilirsiniz:")
    eda_options = [
        "Cinsiyete Göre Hayatta Kalma Oranı (Bar)",
        "Sınıfa Göre Hayatta Kalma Oranı (Bar)",
        "Biniş Limanına Göre Hayatta Kalma Oranı (Bar)",
        "Aile Büyüklüğüne Göre Hayatta Kalma Oranı (Bar)",
        "Yaş Dağılımı (KDE)",
        "Yaş Dağılımı (Histogram)",
        "Fare Dağılımı (Boxplot)",
        "Fare Dağılımı (Violin)",
        "Korelasyon Matrisi (Heatmap)",
        "Cinsiyet Dağılımı (Pie)",
        "Sınıf Dağılımı (Pie)",
        "Fare-Yaş Dağılımı (Scatter)",
        "Cinsiyet ve Sınıfa Göre Hayatta Kalma (Stacked Bar)",
        "Eksik Veri Matrisi (Heatmap)",
        "Pairplot (Temel Değişkenler)",
    ]
    selected = st.multiselect("Grafik Seç:", eda_options, default=eda_options[:3])
    if not selected:
        st.info("Lütfen en az bir grafik seçin.")
        return
    if "Cinsiyete Göre Hayatta Kalma Oranı (Bar)" in selected:
        st.markdown("**Cinsiyete Göre Hayatta Kalma Oranı (%)**")
        fig, ax = plt.subplots()
        data = df.groupby('Sex')['Survived'].mean().reset_index()
        data['Survived'] = data['Survived'] * 100
        sns.barplot(x='Sex', y='Survived', data=data, ci=None, ax=ax, palette='Set2')
        ax.set_ylabel('Hayatta Kalma Oranı (%)')
        for i, v in enumerate(data['Survived']):
            ax.text(i, v + 1, f"%{v:.1f}", ha='center')
        ax.set_title('Cinsiyete Göre Hayatta Kalma Oranı', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Sınıfa Göre Hayatta Kalma Oranı (Bar)" in selected:
        st.markdown("**Sınıfa Göre Hayatta Kalma Oranı (%)**")
        fig, ax = plt.subplots()
        data = df.groupby('Pclass')['Survived'].mean().reset_index()
        data['Survived'] = data['Survived'] * 100
        sns.barplot(x='Pclass', y='Survived', data=data, ci=None, ax=ax, palette='Set1')
        ax.set_ylabel('Hayatta Kalma Oranı (%)')
        for i, v in enumerate(data['Survived']):
            ax.text(i, v + 1, f"%{v:.1f}", ha='center')
        ax.set_title('Sınıfa Göre Hayatta Kalma Oranı', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Biniş Limanına Göre Hayatta Kalma Oranı (Bar)" in selected:
        st.markdown("**Biniş Limanına Göre Hayatta Kalma Oranı (%)**")
        fig, ax = plt.subplots()
        data = df.groupby('Embarked')['Survived'].mean().reset_index()
        data['Survived'] = data['Survived'] * 100
        sns.barplot(x='Embarked', y='Survived', data=data, ci=None, ax=ax, palette='Set3')
        ax.set_ylabel('Hayatta Kalma Oranı (%)')
        for i, v in enumerate(data['Survived']):
            ax.text(i, v + 1, f"%{v:.1f}", ha='center')
        ax.set_title('Biniş Limanına Göre Hayatta Kalma Oranı', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Aile Büyüklüğüne Göre Hayatta Kalma Oranı (Bar)" in selected:
        st.markdown("**Aile Büyüklüğüne Göre Hayatta Kalma Oranı (%)**")
        temp = df.copy()
        temp['FamilySize'] = temp['SibSp'] + temp['Parch'] + 1
        data = temp.groupby('FamilySize')['Survived'].mean().reset_index()
        data['Survived'] = data['Survived'] * 100
        fig, ax = plt.subplots()
        sns.barplot(x='FamilySize', y='Survived', data=data, ci=None, ax=ax, palette='Blues')
        ax.set_ylabel('Hayatta Kalma Oranı (%)')
        for i, v in enumerate(data['Survived']):
            ax.text(i, v + 1, f"%{v:.1f}", ha='center')
        ax.set_title('Aile Büyüklüğüne Göre Hayatta Kalma Oranı', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Yaş Dağılımı (KDE)" in selected:
        st.markdown("**Yaş Dağılımı (KDE)**")
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x='Age', hue='Survived', fill=True, ax=ax, palette='husl')
        ax.set_title('Yaşa Göre Hayatta Kalma Dağılımı', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Yaş Dağılımı (Histogram)" in selected:
        st.markdown("**Yaş Dağılımı (Histogram)**")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30, ax=ax, palette='husl')
        ax.set_title('Yaş Dağılımı (Histogram)', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Fare Dağılımı (Boxplot)" in selected:
        st.markdown("**Fare Dağılımı (Boxplot)**")
        fig, ax = plt.subplots()
        sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=df, ax=ax, palette='Set2')
        ax.set_title('Yolcu Sınıfı ve Bilet Ücretine Göre Hayatta Kalma', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Fare Dağılımı (Violin)" in selected:
        st.markdown("**Fare Dağılımı (Violin Plot)**")
        fig, ax = plt.subplots()
        sns.violinplot(x='Pclass', y='Fare', hue='Survived', data=df, ax=ax, split=True, palette='Set3')
        ax.set_title('Yolcu Sınıfı ve Bilet Ücretine Göre Hayatta Kalma (Violin)', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Korelasyon Matrisi (Heatmap)" in selected:
        st.markdown("**Korelasyon Matrisi (Heatmap)**")
        numeric_df = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('Korelasyon Matrisi', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Cinsiyet Dağılımı (Pie)" in selected:
        st.markdown("**Cinsiyet Dağılımı (Pie Chart, %)**")
        fig, ax = plt.subplots()
        counts = df['Sex'].value_counts(normalize=True) * 100
        counts.plot.pie(autopct='%1.1f%%', ax=ax, colors=['#66b3ff','#ff9999'])
        ax.set_ylabel('')
        ax.set_title('Cinsiyet Dağılımı', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Sınıf Dağılımı (Pie)" in selected:
        st.markdown("**Sınıf Dağılımı (Pie Chart, %)**")
        fig, ax = plt.subplots()
        counts = df['Pclass'].value_counts(normalize=True).sort_index() * 100
        counts.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title('Sınıf Dağılımı', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Fare-Yaş Dağılımı (Scatter)" in selected:
        st.markdown("**Fare-Yaş Dağılımı (Scatter Plot)**")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, ax=ax, palette='husl')
        ax.set_title('Yaş ve Bilet Ücreti Dağılımı', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Cinsiyet ve Sınıfa Göre Hayatta Kalma (Stacked Bar)" in selected:
        st.markdown("**Cinsiyet ve Sınıfa Göre Hayatta Kalma (Stacked Bar)**")
        fig, ax = plt.subplots()
        data = df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack().fillna(0) * 100
        data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_ylabel('Hayatta Kalma Oranı (%)')
        ax.set_title('Cinsiyet ve Sınıfa Göre Hayatta Kalma (Stacked Bar)', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Eksik Veri Matrisi (Heatmap)" in selected:
        st.markdown("**Eksik Veri Matrisi (Heatmap)**")
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
        ax.set_title('Eksik Veri Matrisi', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)
    if "Pairplot (Temel Değişkenler)" in selected:
        st.markdown("**Temel Değişkenler Arası İlişkiler (Pairplot)**")
        st.info("Pairplot büyük veri setlerinde yavaş olabilir. Sadece ilk 200 satır gösteriliyor.")
        fig = sns.pairplot(df[['Age','Fare','Pclass','Survived','SibSp','Parch']].dropna().sample(min(200, len(df))), hue='Survived', palette='husl')
        st.pyplot(fig)
        plt.close('all')


# --- Main Content Area ---
if st.session_state.train_df is not None:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("Original Training Data Sample:")
    st.dataframe(st.session_state.train_df.head())
    st.write(f"Shape: {st.session_state.train_df.shape}")
    
    if st.expander("Show Training Data Description"):
        st.write(st.session_state.train_df.describe(include='all'))

    # Filtre uygula
    df = st.session_state.train_df.copy()
    df = df[df['Sex'].isin(filter_sex)]
    df = df[df['Pclass'].isin(filter_pclass)]
    df = df[(df['Age'] >= filter_age[0]) & (df['Age'] <= filter_age[1])]

    # Gelişmiş EDA grafikleri
    plot_eda_graphs(df)

    if st.button("Show EDA Plots", key="show_eda_plots_main"):
        with st.spinner("Generating EDA plots..."):
            # Ensure predictor.data is set for plot_data_analysis
            if predictor.data is None or not predictor.data.equals(df):
                 predictor.data = df.copy()

            st.subheader("Key Feature Analysis (from `plot_data_analysis`)")
            st.info("The `plot_data_analysis` method in the original script generates multiple plots. For Streamlit, it's better if such methods return figure objects or are called individually. Below are some representative plots.")

            # We'll try to call the main plot_data_analysis. If it uses plt.show() for each, 
            # Streamlit might pick them up. If not, individual plotting calls are better.
            # predictor.plot_data_analysis() # This is likely problematic for Streamlit.
            
            # Replicating some plots manually for better control:
            fig_age, ax_age = plt.subplots()
            sns.kdeplot(data=df, x='Age', hue='Survived', fill=True, ax=ax_age)
            ax_age.set_title('Age Distribution by Survival')
            st.pyplot(fig_age)
            plt.close(fig_age)

            fig_fare, ax_fare = plt.subplots()
            sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=df, ax=ax_fare)
            ax_fare.set_title('Fare Distribution by Pclass and Survival')
            st.pyplot(fig_fare)
            plt.close(fig_fare)

            # Correlation Matrix (on preprocessed data later, or on raw numeric data here)
            st.subheader("Correlation Matrix (Numeric Features of Raw Data)")
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
                ax_corr.set_title('Correlation Matrix of Numeric Features')
                st.pyplot(fig_corr)
                plt.close(fig_corr)
            else:
                st.write("No numeric features to plot correlation matrix for in raw data.")

    st.markdown("---")
    st.write("**Filtrelenmiş veri örneği:**")
    if 'Name' in df.columns:
        df['Name'] = df['Name'].astype(str)
    st.dataframe(df.head())
    st.write(f"Veri şekli: {df.shape}")
    st.info("Sol menüden filtreleri değiştirerek analizleri dinamik olarak güncelleyebilirsiniz.")

    st.header("⚙️ Model Training & Evaluation")
    if st.button("🚀 Train Models & Evaluate", key="train_models_button"):
        if st.session_state.train_df is None or st.session_state.test_df is None or st.session_state.y_train_full is None:
            st.error("Please load training and test data (including 'Survived' in train) first.")
        else:
            with st.spinner("Preprocessing data and training models... This may take a few minutes."):
                try:
                    train_df_features = st.session_state.train_df.drop('Survived', axis=1)
                    st.write("Preprocessing training data...")
                    st.session_state.processed_train_df = predictor.preprocess_data(train_df_features.copy(), is_train=True)
                    st.write(f"Processed training data shape: {st.session_state.processed_train_df.shape}")
                    st.write("Preprocessing test data...")
                    st.session_state.processed_test_df = predictor.preprocess_data(st.session_state.test_df.copy(), is_train=False)
                    st.write(f"Processed test data shape: {st.session_state.processed_test_df.shape}")
                    st.session_state.features_for_scaling = [col for col in st.session_state.processed_train_df.columns if col not in ['Survived', 'PassengerId']]
                    X_train_processed = st.session_state.processed_train_df[st.session_state.features_for_scaling].copy()
                    X_test_aligned = pd.DataFrame(columns=X_train_processed.columns)
                    for col in X_train_processed.columns:
                        if col in st.session_state.processed_test_df.columns:
                            X_test_aligned[col] = st.session_state.processed_test_df[col]
                        else:
                            st.warning(f"Warning: Column '{col}' not in processed test set, filling with 0.")
                            X_test_aligned[col] = 0
                    X_test_processed_aligned = X_test_aligned[X_train_processed.columns]
                    st.write("Scaling data...")
                    X_train_scaled_array = predictor.scaler.fit_transform(X_train_processed)
                    X_test_scaled_array = predictor.scaler.transform(X_test_processed_aligned)
                    st.session_state.X_train_scaled_df = pd.DataFrame(X_train_scaled_array, columns=st.session_state.features_for_scaling)
                    st.session_state.X_test_scaled = X_test_scaled_array

                    # --- Model Training ---
                    st.write(f"Model seçimi: {selected_model}")
                    if selected_model == "Otomatik (En iyi CV F1)":
                        st.info("Tüm modeller eğitiliyor ve karşılaştırılıyor... Bu işlem biraz zaman alabilir.")
                        
                        # Define models and their parameters
                        models = {
                            "Logistic Regression": (LogisticRegression(random_state=42), 
                                {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear'], 'max_iter': [1000]}),
                            "Decision Tree": (DecisionTreeClassifier(random_state=42), 
                                {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}),
                            "KNN": (KNeighborsClassifier(), 
                                {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
                            "Random Forest": (RandomForestClassifier(random_state=42), 
                                {'n_estimators': [100], 'max_depth': [5, 7], 'min_samples_split': [2], 'min_samples_leaf': [1]}),
                            "XGBoost": (XGBClassifier(random_state=42, eval_metric='logloss'), 
                                {'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.1], 'subsample': [1.0], 'colsample_bytree': [1.0]})
                        }
                        
                        # Train and evaluate all models
                        results = []
                        best_score = -1
                        best_model_name = None
                        best_model = None
                        
                        for model_name, (base_model, params) in models.items():
                            st.write(f"Eğitiliyor: {model_name}...")
                            gs = GridSearchCV(base_model, params, cv=3, scoring='f1_macro', refit=True, n_jobs=-1)
                            gs.fit(st.session_state.X_train_scaled_df, st.session_state.y_train_full)
                            
                            score = gs.best_score_
                            results.append({
                                'Model': model_name,
                                'CV_F1_macro': score,
                                'CV_Accuracy': gs.cv_results_['mean_test_score'][gs.best_index_]
                            })
                            
                            if score > best_score:
                                best_score = score
                                best_model_name = model_name
                                best_model = gs.best_estimator_
                        
                        # Store results
                        st.session_state.model_performance_results = results
                        predictor.models = {name: model for name, (model, _) in models.items()}
                        predictor.best_model = best_model
                        st.session_state.best_model_name = best_model_name
                        st.session_state.model_trained = True
                        
                        # Show results
                        st.success(f"En iyi model: {best_model_name} (CV F1-macro: {best_score:.4f})")
                        st.write("Tüm modellerin performans karşılaştırması:")
                        perf_df = pd.DataFrame(results)
                        st.dataframe(perf_df.sort_values('CV_F1_macro', ascending=False))
                        
                    else:
                        st.info(f"Sadece {selected_model} eğitiliyor...")
                        from sklearn.model_selection import GridSearchCV
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.tree import DecisionTreeClassifier
                        from sklearn.neighbors import KNeighborsClassifier
                        from sklearn.ensemble import RandomForestClassifier
                        from xgboost import XGBClassifier
                        
                        model_map = {
                            "Logistic Regression": (LogisticRegression(random_state=42), {'C': [1.0], 'solver': ['liblinear'], 'max_iter': [1000]}),
                            "Decision Tree": (DecisionTreeClassifier(random_state=42), {'max_depth': [5], 'min_samples_split': [2], 'min_samples_leaf': [1]}),
                            "KNN": (KNeighborsClassifier(), {'n_neighbors': [5], 'weights': ['uniform']}),
                            "Random Forest": (RandomForestClassifier(random_state=42), {'n_estimators': [100], 'max_depth': [7], 'min_samples_split': [2], 'min_samples_leaf': [1]}),
                            "XGBoost": (XGBClassifier(random_state=42, eval_metric='logloss'), {'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.1], 'subsample': [1.0], 'colsample_bytree': [1.0]}),
                        }
                        
                        if selected_model in model_map:
                            base_model, params = model_map[selected_model]
                            gs = GridSearchCV(base_model, params, cv=3, scoring='f1_macro', refit=True, n_jobs=-1)
                            st.info(f"{selected_model} için GridSearch başlatıldı...")
                            gs.fit(st.session_state.X_train_scaled_df, st.session_state.y_train_full)
                            predictor.models = {selected_model: gs}
                            predictor.best_model = gs.best_estimator_
                            st.session_state.best_model_name = selected_model
                            st.session_state.model_trained = True
                            st.session_state.model_performance_results = [{
                                'Model': selected_model,
                                'CV_F1_macro': gs.best_score_,
                                'CV_Accuracy': gs.cv_results_['mean_test_score'][gs.best_index_] if 'mean_test_score' in gs.cv_results_ else 0
                            }]
                            st.success(f"{selected_model} başarıyla eğitildi!")
                        else:
                            st.error(f"{selected_model} için otomatik eğitim desteklenmiyor. Lütfen Otomatik veya temel modellerden birini seçin.")
                            st.session_state.model_trained = False
                            st.session_state.model_performance_results = []
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during model training: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    st.session_state.model_trained = False
                    st.session_state.model_performance_results = []
                    st.rerun()

    if st.session_state.model_trained:
        st.subheader("Model Performance Summary")
        if st.session_state.model_performance_results:
            perf_df = pd.DataFrame(st.session_state.model_performance_results)
            st.dataframe(perf_df[['Model', 'CV_F1_macro', 'CV_Accuracy']].sort_values(by='CV_F1_macro', ascending=False).reset_index(drop=True))

            # Plot Model Comparison (CV F1)
            fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
            plot_df_perf = perf_df[perf_df['CV_F1_macro'] > 0].sort_values('CV_F1_macro', ascending=False)
            if not plot_df_perf.empty:
                sns.barplot(x='CV_F1_macro', y='Model', data=plot_df_perf, palette='viridis', ax=ax_comp, orient='h')
                ax_comp.set_title('Models CV F1-macro Scores')
                ax_comp.set_xlabel('CV F1-macro Score')
                ax_comp.set_ylabel('Model')
                st.pyplot(fig_comp)
            else:
                st.write("No models with CV F1 > 0 to plot.")
            plt.close(fig_comp)

        # Feature Importance for Best Model
        if predictor.best_model and st.session_state.best_model_name and hasattr(predictor.best_model, 'feature_importances_') and st.session_state.X_train_scaled_df is not None:
            st.subheader(f"Feature Importances for {st.session_state.best_model_name}")
            importances = pd.DataFrame({
                'feature': st.session_state.X_train_scaled_df.columns,
                'importance': predictor.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_imp, ax_imp = plt.subplots(figsize=(10, min(8, len(importances)*0.3 + 1))) # Adjust height
            sns.barplot(x='importance', y='feature', data=importances.head(20), ax=ax_imp, orient='h')
            ax_imp.set_title(f'Top 20 Feature Importances ({st.session_state.best_model_name})')
            st.pyplot(fig_imp)
            plt.close(fig_imp)
        
        # Learning Curve for Best Model
        if predictor.best_model and st.session_state.best_model_name and st.session_state.X_train_scaled_df is not None and st.session_state.y_train_full is not None:
            st.subheader(f"Learning Curve for {st.session_state.best_model_name}")
            if st.button(f"Show Learning Curve for {st.session_state.best_model_name}", key="lc_button"):
                with st.spinner(f"Generating Learning Curve..."):
                    try:
                        # plot_learning_curve in titanic_model.py returns `plt` module.
                        # st.pyplot(plt) or st.pyplot(plt.gcf()) should work.
                        estimator_for_lc = clone(predictor.best_model)
                        # The plot_learning_curve function needs to be callable and use plt to draw.
                        # It's assumed it draws to the current global figure.
                        
                        # Create a figure here for the method to plot on if it takes an ax
                        fig_lc_obj, ax_lc_obj = plt.subplots(figsize=(10,6))
                        
                        # Call the predictor's method. It's expected to plot on the current axes or a new one.
                        # And it should return the plt object or the figure object.
                        # If it returns plt module:
                        returned_plt_obj = predictor.plot_learning_curve(
                                                    estimator_for_lc,
                                                    f"Learning Curve ({st.session_state.best_model_name})",
                                                    st.session_state.X_train_scaled_df,
                                                    st.session_state.y_train_full,
                                                    axes=ax_lc_obj, # Pass axes if function supports it
                                                    cv=3, n_jobs=-1) # Reduced CV for speed
                        
                        st.pyplot(fig_lc_obj) # Display the figure we created and passed
                        plt.close(fig_lc_obj) # Close local figure
                        
                    except Exception as e:
                        st.error(f"Could not plot learning curve: {e}")
                        st.error(traceback.format_exc())

        st.header("🧪 Make Predictions (on Test Data)")
        if st.session_state.X_test_scaled is not None and predictor.best_model:
            if st.button("Predict on Test Set & Generate Submission File", key="pred_test_button"):
                with st.spinner("Making predictions on the test set..."):
                    test_predictions = predictor.best_model.predict(st.session_state.X_test_scaled)
                    submission_df = pd.DataFrame({
                        'PassengerId': st.session_state.test_passenger_ids,
                        'Survived': test_predictions.astype(int)
                    })
                    st.dataframe(submission_df.head())
                    
                    csv = submission_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Submission.csv",
                        data=csv,
                        file_name="submission.csv",
                        mime="text/csv",
                    )
                    st.success("Submission file generated and ready for download.")
        
        st.header("🧠 Model Interpretation with SHAP")
        if predictor.best_model and st.session_state.best_model_name and st.session_state.X_test_scaled is not None and st.session_state.features_for_scaling is not None:
            # Check if the model supports SHAP analysis
            shap_supported_models = ["Random Forest", "XGBoost", "Decision Tree"]
            if st.session_state.best_model_name in shap_supported_models:
                if st.button(f"Run SHAP Analysis for {st.session_state.best_model_name}", key="shap_button"):
                    with st.spinner("Calculating SHAP values... This can take time."):
                        try:
                            best_model_for_shap = predictor.best_model
                            
                            # Prepare background data
                            if st.session_state.X_train_scaled_df.shape[0] > 50:
                                background_data = shap.sample(st.session_state.X_train_scaled_df, 50)
                            else:
                                background_data = st.session_state.X_train_scaled_df.copy()
                            
                            # Handle NaN values in background data
                            background_data = background_data.fillna(background_data.mean())
                            
                            # Prepare test data for SHAP
                            X_test_scaled_df_for_shap = pd.DataFrame(st.session_state.X_test_scaled, columns=st.session_state.features_for_scaling)
                            if X_test_scaled_df_for_shap.shape[0] > 100:
                                X_test_scaled_df_for_shap = X_test_scaled_df_for_shap.sample(100, random_state=42)
                            
                            # Handle NaN values in test data
                            X_test_scaled_df_for_shap = X_test_scaled_df_for_shap.fillna(X_test_scaled_df_for_shap.mean())

                            # Use TreeExplainer for tree-based models
                            explainer = shap.TreeExplainer(best_model_for_shap, data=background_data)
                            
                            # Calculate SHAP values
                            shap_values = explainer.shap_values(X_test_scaled_df_for_shap)

                            # SHAP plots
                            st.subheader("SHAP Summary Plot (Bar)")
                            fig_shap_bar, ax_shap_bar = plt.subplots()
                            # For binary classification, shap_values can be a list [shap_values_class0, shap_values_class1]
                            shap_val_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
                            shap.summary_plot(shap_val_to_plot, X_test_scaled_df_for_shap, plot_type="bar", show=False)
                            st.pyplot(plt.gcf())
                            plt.close(fig_shap_bar)

                            st.subheader("SHAP Summary Plot (Dot/Violin)")
                            fig_shap_dot, ax_shap_dot = plt.subplots()
                            shap.summary_plot(shap_val_to_plot, X_test_scaled_df_for_shap, show=False)
                            st.pyplot(plt.gcf())
                            plt.close(fig_shap_dot)
                            
                        except Exception as e:
                            st.error(f"Error during SHAP analysis: {str(e)}")
                            import traceback
                            st.error("Detailed error information:")
                            st.code(traceback.format_exc())
            else:
                st.info(f"SHAP analizi şu anda {st.session_state.best_model_name} modeli için desteklenmiyor. SHAP analizi sadece ağaç tabanlı modeller (Random Forest, XGBoost, Decision Tree) için kullanılabilir.")

        st.header("🙋 Individual Passenger Prediction")
        if predictor.best_model and st.session_state.features_for_scaling:
            st.write(f"Using model: **{st.session_state.best_model_name}** for prediction.")
            
            with st.form(key="individual_prediction_form"):
                st.subheader("Enter Passenger Details:")
                cols_form = st.columns(3)
                input_data = {}
                with cols_form[0]:
                    input_data['Name'] = st.text_input("Name (e.g., Smith, Mr. John)", "Doe, Mr. John")
                    input_data['Pclass'] = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=0)
                    input_data['Sex'] = st.selectbox("Sex", ['male', 'female'], index=1)
                with cols_form[1]:
                    input_data['Age'] = st.number_input("Age", min_value=0.1, max_value=100.0, value=30.0, step=0.1)
                    input_data['SibSp'] = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
                    input_data['Parch'] = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
                with cols_form[2]:
                    input_data['Ticket'] = st.text_input("Ticket Number", "A/5 21171")
                    input_data['Fare'] = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=30.0, step=0.1)
                    input_data['Cabin'] = st.text_input("Cabin (or 'NaN')", "NaN")
                    input_data['Embarked'] = st.selectbox("Port of Embarkation (Embarked)", ['S', 'C', 'Q'], index=0)
                
                submit_button = st.form_submit_button(label='Predict Survival')

            if submit_button:
                if input_data['Cabin'].strip().upper() == 'NAN':
                    input_data['Cabin'] = np.nan
                
                passenger_df = pd.DataFrame([input_data])
                with st.spinner("Processing input and predicting..."):
                    try:
                        # Preprocess using is_train=False (uses fitted encoders/imputers from predictor)
                        processed_passenger_df = predictor.preprocess_data(passenger_df.copy(), is_train=False)
                        
                        # Align columns with features_for_scaling
                        aligned_passenger_data = {}
                        for col in st.session_state.features_for_scaling:
                            if col in processed_passenger_df.columns:
                                aligned_passenger_data[col] = processed_passenger_df[col].iloc[0]
                            else:
                                st.warning(f"Feature '{col}' (expected from training) not found in processed input. Defaulting to 0.")
                                aligned_passenger_data[col] = 0
                        
                        aligned_passenger_df_final = pd.DataFrame([aligned_passenger_data], columns=st.session_state.features_for_scaling)
                        
                        # Handle any remaining NaN values
                        aligned_passenger_df_final = aligned_passenger_df_final.fillna(0)
                        
                        # Scale using the fitted scaler
                        passenger_scaled = predictor.scaler.transform(aligned_passenger_df_final)
                        
                        # Make prediction
                        prediction = predictor.best_model.predict(passenger_scaled)
                        prediction_proba = predictor.best_model.predict_proba(passenger_scaled)
                        
                        if prediction[0] == 1:
                            st.success(f"### Predicted: **Survived** (Confidence: {prediction_proba[0][1]*100:.1f}%)")
                        else:
                            st.error(f"### Predicted: **Did Not Survive** (Confidence: {prediction_proba[0][0]*100:.1f}%)")
                        
                        if st.checkbox("Show processed data for this passenger?"):
                            st.write("Original Input:")
                            st.dataframe(passenger_df)
                            st.write("Processed Features (before scaling):")
                            st.dataframe(processed_passenger_df)
                            st.write("Scaled Features (input to model):")
                            st.dataframe(pd.DataFrame(passenger_scaled, columns=st.session_state.features_for_scaling))

                    except Exception as e:
                        st.error(f"Error during individual prediction: {str(e)}")
                        st.error(traceback.format_exc())
    else:
        st.info("Train a model first (using the button in 'Model Training & Evaluation' section) to enable individual predictions and other analyses.")

else:
    st.info("👋 Welcome! Please load data using the sidebar controls to begin the analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown("Streamlit app by AI, based on `titanic_model.py`.") 
