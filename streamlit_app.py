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


# --- Sidebar ---
st.sidebar.title("🚢 Titanic Survival Prediction")
st.sidebar.markdown("""
Bu uygulama, Titanic yolcularının hayatta kalıp kalamayacağını makine öğrenmesi ile tahmin eder. 
Veri analizi, model karşılaştırma ve bireysel tahmin özellikleri sunar.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Streamlit app by AI, based on `titanic_model.py`.")

# --- Main Page Akışı ---
if 'app_stage' not in st.session_state:
    st.session_state.app_stage = 'start'

st.title("🚢 Titanic Survival Prediction Interactive Dashboard")

if st.session_state.app_stage == 'start':
    st.markdown("""
    <div style='display: flex; justify-content: center; align-items: center; height: 300px;'>
        <button style='font-size:2rem;padding:2rem 4rem;background:#4F8BF9;color:white;border:none;border-radius:1rem;cursor:pointer;' onclick="window.location.reload()">Start</button>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start", key="start_button", help="Veriyi yükle ve uygulamayı başlat"):
        # Varsayılan dosyaları yükle
        predictor = get_predictor_instance()
        train_df_loaded = predictor.load_data('train (2).csv')
        test_df_loaded = predictor.load_data('test (2).csv')
        if train_df_loaded is not None and test_df_loaded is not None:
            st.session_state.train_df = train_df_loaded
            st.session_state.test_df = test_df_loaded
            if 'Survived' in train_df_loaded.columns:
                st.session_state.y_train_full = train_df_loaded['Survived'].copy()
                predictor.data = st.session_state.train_df.copy()
            if 'PassengerId' in test_df_loaded.columns:
                st.session_state.test_passenger_ids = test_df_loaded['PassengerId'].copy()
            else:
                st.session_state.test_passenger_ids = pd.Series(range(1, len(test_df_loaded) + 1), name="PassengerId")
            st.session_state.predictor = predictor
            st.session_state.app_stage = 'eda'
            st.experimental_rerun()
        else:
            st.error("Veri yüklenemedi. Lütfen dosyaları kontrol edin.")

elif st.session_state.app_stage == 'eda':
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("Veri başarıyla yüklendi. Aşağıda örnek satırlar:")
    st.dataframe(st.session_state.train_df.head())
    st.write(f"Veri şekli: {st.session_state.train_df.shape}")
    if st.button("Show All Plots", key="show_all_plots_button", help="Tüm önemli EDA grafiklerini göster"):
        st.session_state.show_all_plots = True
    if st.session_state.get('show_all_plots', False):
        st.info("Tüm önemli EDA grafiklerini aşağıda görebilirsiniz.")
        # Tüm önemli EDA grafiklerini otomatik ve okunur şekilde göster
        plot_eda_graphs(st.session_state.train_df)
    if st.button("Train & Test Models", key="train_test_models_button", help="Model eğitimi ve test işlemlerini başlat"):
        st.session_state.app_stage = 'model'
        st.experimental_rerun()

elif st.session_state.app_stage == 'model':
    st.header("⚙️ Model Training & Evaluation")
    # Model eğitim ve test bölümü (mevcut kodun ilgili kısmı)
    # ... mevcut model eğitim kodunu buraya taşı ...
    # (Kısa tutmak için, yukarıdaki model eğitim kodunu buraya taşıyın)
    # Kullanıcı model eğitimi ve testini burada başlatacak.
    # Model eğitimi tamamlanınca sonuçlar ve test predictionları gösterilecek.
    # (Mevcut kodun model eğitimi ve test bölümü buraya taşınacak.)
    # Ayrıca, kullanıcı isterse tekrar EDA'ya dönebilsin diye bir buton ekleyin.
    if st.button("Back to EDA", key="back_to_eda_button"):
        st.session_state.app_stage = 'eda'
        st.experimental_rerun()

# --- plot_eda_graphs fonksiyonu ---
def plot_eda_graphs(df):
    # Tüm önemli EDA grafiklerini otomatik ve okunur şekilde göster
    st.subheader("Cinsiyete Göre Hayatta Kalma Oranı (%)")
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

    st.subheader("Sınıfa Göre Hayatta Kalma Oranı (%)")
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

    st.subheader("Biniş Limanına Göre Hayatta Kalma Oranı (%)")
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

    st.subheader("Aile Büyüklüğüne Göre Hayatta Kalma Oranı (%)")
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

    st.subheader("Yaş Dağılımı (KDE)")
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x='Age', hue='Survived', fill=True, ax=ax, palette='husl')
    ax.set_title('Yaşa Göre Hayatta Kalma Dağılımı', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Yaş Dağılımı (Histogram)")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30, ax=ax, palette='husl')
    ax.set_title('Yaş Dağılımı (Histogram)', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Fare Dağılımı (Boxplot)")
    fig, ax = plt.subplots()
    sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=df, ax=ax, palette='Set2')
    ax.set_title('Yolcu Sınıfı ve Bilet Ücretine Göre Hayatta Kalma', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Fare Dağılımı (Violin Plot)")
    fig, ax = plt.subplots()
    sns.violinplot(x='Pclass', y='Fare', hue='Survived', data=df, ax=ax, split=True, palette='Set3')
    ax.set_title('Yolcu Sınıfı ve Bilet Ücretine Göre Hayatta Kalma (Violin)', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Korelasyon Matrisi (Heatmap)")
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Korelasyon Matrisi', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Cinsiyet Dağılımı (Pie Chart, %)")
    fig, ax = plt.subplots()
    counts = df['Sex'].value_counts(normalize=True) * 100
    counts.plot.pie(autopct='%1.1f%%', ax=ax, colors=['#66b3ff','#ff9999'])
    ax.set_ylabel('')
    ax.set_title('Cinsiyet Dağılımı', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sınıf Dağılımı (Pie Chart, %)")
    fig, ax = plt.subplots()
    counts = df['Pclass'].value_counts(normalize=True).sort_index() * 100
    counts.plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    ax.set_title('Sınıf Dağılımı', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Fare-Yaş Dağılımı (Scatter Plot)")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, ax=ax, palette='husl')
    ax.set_title('Yaş ve Bilet Ücreti Dağılımı', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Cinsiyet ve Sınıfa Göre Hayatta Kalma (Stacked Bar)")
    fig, ax = plt.subplots()
    data = df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack().fillna(0) * 100
    data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_ylabel('Hayatta Kalma Oranı (%)')
    ax.set_title('Cinsiyet ve Sınıfa Göre Hayatta Kalma (Stacked Bar)', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Eksik Veri Matrisi (Heatmap)")
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
    ax.set_title('Eksik Veri Matrisi', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Temel Değişkenler Arası İlişkiler (Pairplot)")
    st.info("Pairplot büyük veri setlerinde yavaş olabilir. Sadece ilk 200 satır gösteriliyor.")
    fig = sns.pairplot(df[['Age','Fare','Pclass','Survived','SibSp','Parch']].dropna().sample(min(200, len(df))), hue='Survived', palette='husl')
    st.pyplot(fig)
    plt.close('all') 
