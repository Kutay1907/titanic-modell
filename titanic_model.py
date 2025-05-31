# -*- coding: utf-8 -*-
"""
# Titanic Survival Prediction Model
Bu notebook Titanic veri seti üzerinde çeşitli makine öğrenmesi modellerini karşılaştırmaktadır.
"""

# Gerekli kütüphaneleri içe aktar
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import warnings
import shap # SHAP kütüphanesini içe aktar
from sklearn.base import clone # For cloning estimators for learning/validation curves
import joblib # Model ve nesneleri kaydetmek/yüklemek için

warnings.filterwarnings('ignore')

# Görselleştirme ayarları
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Hiçbiri yoksa varsayılanı kullan
sns.set_palette("husl")

"""## Model Sınıfını Tanımla"""

# HÜCRE 5: TitanicPredictor sınıfını tanımla
class TitanicPredictor:
    def __init__(self):
        self.data = None
        self.models = {
            'Logistic Regression': None,
            'Decision Tree': None,
            'KNN': None,
            'Random Forest': None,
            'XGBoost': None
        }
        self.base_models_for_stacking = [] # Stacking için temel modelleri tutacak liste
        self.ensemble_model_cv_scores = {} # Ensemble modellerin CV skorlarını saklamak için
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False) # Polinomik özellikler için
        # Her kategorik özellik için ayrı LabelEncoder
        self.label_encoders = {}
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.best_model = None
        self.best_accuracy = 0
        # Bilet öneki için eğitilmiş encoder'ı saklamak için
        self.ticket_prefix_encoder = None
        
    def load_data(self, filepath):
        """Belirtilen yoldan veri setini yükle ve self.data'yı GÜNCELLEMEZ."""
        try:
            data = pd.read_csv(filepath)
            print(f"'{filepath}' başarıyla yüklendi! Toplam {len(data)} satır veri bulundu.")
            return data
        except FileNotFoundError:
            print(f"HATA: '{filepath}' konumunda veri seti bulunamadı!")
            print("Lütfen dosya yolunu kontrol edin.")
            return None
        except Exception as e:
            print(f"HATA: Veri seti yüklenirken bir hata oluştu: {str(e)}")
            return None
    
    def create_title_feature(self, df):
        """İsimlerden unvan özelliği çıkar"""
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
            'Dr': 5, 'Rev': 5, 'Col': 5, 'Major': 5, 'Mlle': 2,
            'Countess': 3, 'Ms': 2, 'Lady': 3, 'Jonkheer': 1,
            'Don': 1, 'Mme': 3, 'Capt': 5, 'Sir': 5
        }
        df['Title'] = df['Title'].map(title_mapping)
        return df

    def create_family_features(self, df):
        """Aile ile ilgili özellikler oluştur"""
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 1
        df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
        df['FamilyType'] = pd.cut(df['FamilySize'], 
                                 bins=[0, 1, 4, float('inf')], 
                                 labels=[0, 1, 2])
        return df

    def create_age_features(self, df):
        """Yaş ile ilgili özellikler oluştur"""
        df['AgeGroup'] = pd.cut(df['Age'], 
                               bins=[-float('inf'), 16, 32, 48, 64, float('inf')],
                               labels=[0, 1, 2, 3, 4])
        df['Age*Class'] = df['Age'] * df['Pclass']
        return df

    def create_fare_features(self, df):
        """Bilet ücreti ile ilgili özellikler oluştur"""
        # FareGroup için qcut kullanılırken duplicate bin edge hatasını önlemek için duplicates='drop' ekle
        df['FareGroup'] = pd.qcut(df['Fare'].fillna(df['Fare'].median()), 4, labels=False, duplicates='drop')
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']
        df['Fare*Class'] = df['Fare'] * df['Pclass']
        return df

    def create_ticket_features(self, df, is_train=True):
        """Biletlerden özellik çıkar"""
        # Bilet Frekansı
        df['TicketFrequency'] = df.groupby('Ticket')['Ticket'].transform('count')

        # Bilet Öneki
        df['TicketPrefix'] = df['Ticket'].apply(lambda x: 'NONE' if str(x).isdigit() else str(x).replace('.','').replace('/','').strip().split(' ')[0].upper())
        
        # Önekleri daha da basitleştir (sayısal ve çok kısa olanları birleştir)
        df['TicketPrefix'] = df['TicketPrefix'].apply(lambda x: 'OTHER' if len(x) < 2 and x != 'C' and x != 'S' and x != 'P' else x)
        
        # Nadir görülen önekleri 'OTHER' olarak grupla (eğitim setine göre)
        if self.ticket_prefix_encoder is None: # Eğitim aşaması
            prefix_counts = df['TicketPrefix'].value_counts()
            rare_prefixes = prefix_counts[prefix_counts < 10].index
            df['TicketPrefix'] = df['TicketPrefix'].replace(rare_prefixes, 'OTHER')
            self.ticket_prefix_encoder = LabelEncoder()
            df['TicketPrefix_Code'] = self.ticket_prefix_encoder.fit_transform(df['TicketPrefix'])
        else: # Test/tahmin aşaması
            # Eğitimde görülmeyen yeni prefixleri 'OTHER' yap
            known_prefixes = self.ticket_prefix_encoder.classes_
            df['TicketPrefix'] = df['TicketPrefix'].apply(lambda x: 'OTHER' if x not in known_prefixes else x)
            df['TicketPrefix_Code'] = self.ticket_prefix_encoder.transform(df['TicketPrefix'])
            
        return df

    def handle_missing_values(self, df, is_train=True):
        """Eksik değerleri akıllıca doldur. is_train=True ise imputer'ları fit eder."""
        # Kategorik değişkenler için
        categorical_cols = [col for col in ['Embarked', 'Cabin'] if col in df.columns]
        if categorical_cols:
            if is_train:
                df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
            else:
                # Sadece eğitimde görülen ve testte de olan sütunları transform et
                transform_cat_cols = [col for col in categorical_cols if col in self.categorical_imputer.feature_names_in_]
                if transform_cat_cols:
                    df[transform_cat_cols] = self.categorical_imputer.transform(df[transform_cat_cols])
        
        # Sayısal değişkenler için
        numeric_cols = [col for col in ['Age', 'Fare'] if col in df.columns] # Cabin_Number eklenecekse buraya dahil edilebilir
        if numeric_cols:
            if is_train:
                df[numeric_cols] = self.numeric_imputer.fit_transform(df[numeric_cols])
            else:
                transform_num_cols = [col for col in numeric_cols if col in self.numeric_imputer.feature_names_in_]
                if transform_num_cols:
                    df[transform_num_cols] = self.numeric_imputer.transform(df[transform_num_cols])
        return df

    def preprocess_data(self, df, is_train=True):
        """Tüm veri ön işleme adımlarını uygula. is_train=True ise encoder/scaler vb. fit eder."""
        original_ticket = df['Ticket'].copy() if 'Ticket' in df.columns else None
        # PassengerId'yi test için sakla, index resetlemeden önce.
        passenger_id_series_for_test = None
        if not is_train and 'PassengerId' in df.columns:
            passenger_id_series_for_test = df['PassengerId'].copy()

        # Ana df'in kopyasını al ve index'i resetle
        current_df = df.copy()
        current_df.reset_index(drop=True, inplace=True)

        # Adım 1: Eksik Değer Doldurma (handle_missing_values ile)
        current_df = self.handle_missing_values(current_df, is_train=is_train) # handle_missing_values da bir kopya üzerinde çalışabilir veya inplace yapabilir
        current_df.reset_index(drop=True, inplace=True) # handle_missing_values sonrası da index'i garantiye al
        
        # Adım 2: Temel Özellik Mühendisliği
        if 'Name' in current_df.columns: current_df = self.create_title_feature(current_df)
        current_df = self.create_family_features(current_df)
        if 'Age' in current_df.columns and 'Pclass' in current_df.columns: current_df = self.create_age_features(current_df)
        if 'Fare' in current_df.columns and 'FamilySize' in current_df.columns and 'Pclass' in current_df.columns: current_df = self.create_fare_features(current_df)

        # Adım 3: Bilet Özellikleri
        if original_ticket is not None:
            # original_ticket'ın index'i current_df ile eşleşmeyebilir. Değerleri atamadan önce index resetle.
            ticket_series = original_ticket.reset_index(drop=True)
            current_df['Ticket'] = ticket_series
            current_df = self.create_ticket_features(current_df, is_train=is_train)
        elif 'Ticket' in current_df.columns:
            current_df = self.create_ticket_features(current_df, is_train=is_train)
        else: 
            current_df['TicketFrequency'] = 1
            current_df['TicketPrefix_Code'] = 0 
            if is_train and self.ticket_prefix_encoder is None:
                self.ticket_prefix_encoder = LabelEncoder()
                self.ticket_prefix_encoder.fit(['NONE', 'OTHER'])

        # Adım 4: Kabin Özellikleri
        current_df['Has_Cabin'] = current_df['Cabin'].notna().astype(int) if 'Cabin' in current_df.columns else 0
        current_df['Cabin_Letter'] = current_df['Cabin'].str.slice(0, 1).fillna('U') if 'Cabin' in current_df.columns else 'U'
        
        cabin_mapping = {
            'A': 'Top_Deck', 'B': 'Top_Deck', 'C': 'Middle_Deck',
            'D': 'Middle_Deck', 'E': 'Lower_Deck', 'F': 'Lower_Deck',
            'G': 'Lower_Deck', 'T': 'Staff', 'U': 'Unknown'
        }
        current_df['Cabin_Location'] = current_df['Cabin_Letter'].map(cabin_mapping).fillna('Unknown')

        if is_train:
            self.label_encoders['Cabin_Location'] = LabelEncoder()
            current_df['Cabin_Location_Code'] = self.label_encoders['Cabin_Location'].fit_transform(current_df['Cabin_Location'])
        else:
            known_cabin_locations = self.label_encoders['Cabin_Location'].classes_ if 'Cabin_Location' in self.label_encoders else ['Unknown']
            current_df['Cabin_Location'] = current_df['Cabin_Location'].apply(lambda x: x if x in known_cabin_locations else 'Unknown')
            if 'Cabin_Location' in self.label_encoders:
                current_df['Cabin_Location_Code'] = self.label_encoders['Cabin_Location'].transform(current_df['Cabin_Location'])
            else: 
                le_temp = LabelEncoder()
                current_df['Cabin_Location_Code'] = le_temp.fit_transform(current_df['Cabin_Location']) 

        current_df['Cabin_Number'] = current_df['Cabin'].str.extract(r'([0-9]+)').astype(float) if 'Cabin' in current_df.columns else np.nan
        if is_train:
            self.cabin_number_median = current_df['Cabin_Number'].median()
            current_df['Cabin_Number'].fillna(self.cabin_number_median, inplace=True)
        else:
            median_to_fill = self.cabin_number_median if hasattr(self, 'cabin_number_median') and pd.notna(self.cabin_number_median) else (current_df['Cabin_Number'].median() if 'Cabin_Number' in current_df.columns and not current_df['Cabin_Number'].empty else 0)
            current_df['Cabin_Number'].fillna(median_to_fill, inplace=True)
        current_df['Cabin_Number'] = current_df['Cabin_Number'].fillna(0)

        # Adım 5: Kategorik Kodlamalar
        current_df['Sex_Code'] = current_df['Sex'].map({'female': 0, 'male': 1}) if 'Sex' in current_df.columns else 0.5

        embarked_col_present = 'Embarked' in current_df.columns
        if is_train:
            if embarked_col_present:
                self.embarked_mode = current_df['Embarked'].mode()[0] if not current_df['Embarked'].dropna().empty else 'S'
                current_df['Embarked'].fillna(self.embarked_mode, inplace=True)
                self.label_encoders['Embarked'] = LabelEncoder().fit(current_df['Embarked'])
                current_df['Embarked_Code'] = self.label_encoders['Embarked'].transform(current_df['Embarked'])
            else: 
                current_df['Embarked_Code'] = 0 
                self.label_encoders['Embarked'] = None 
        else:
            if embarked_col_present:
                current_embarked_mode = self.embarked_mode if hasattr(self, 'embarked_mode') else (current_df['Embarked'].mode()[0] if not current_df['Embarked'].dropna().empty else 'S')
                current_df['Embarked'].fillna(current_embarked_mode, inplace=True)
                if self.label_encoders.get('Embarked'): 
                    known_embarked = self.label_encoders['Embarked'].classes_
                    current_df['Embarked'] = current_df['Embarked'].apply(lambda x: x if str(x) in known_embarked else current_embarked_mode)
                    current_df['Embarked_Code'] = self.label_encoders['Embarked'].transform(current_df['Embarked'])
                else: 
                    le_temp = LabelEncoder()
                    current_df['Embarked_Code'] = le_temp.fit_transform(current_df['Embarked'])
            else: 
                current_df['Embarked_Code'] = 0
        
        if 'Title' in current_df.columns:
            current_df['Title_Code'] = current_df['Title']
        else: 
            current_df['Title_Code'] = 1 

        # Adım 6: Polinomik Özellikler
        poly_cols = ['Age', 'Fare']
        temp_poly_df_list = []
        for col in poly_cols:
            if col not in current_df.columns:
                 default_val = getattr(self.numeric_imputer, 'statistics_', [0,0])[poly_cols.index(col)] if hasattr(self.numeric_imputer, 'statistics_') and hasattr(self.numeric_imputer, 'feature_names_in_') and col in self.numeric_imputer.feature_names_in_ and len(getattr(self.numeric_imputer, 'statistics_', [])) == len(poly_cols) else 0
                 temp_poly_df_list.append(pd.Series(np.full(len(current_df), default_val), name=col, index=current_df.index))
            else:
                 temp_poly_df_list.append(current_df[col])
        
        df_for_poly = pd.concat(temp_poly_df_list, axis=1)

        if is_train:
            poly_features_transformed = self.poly.fit_transform(df_for_poly)
        else:
            poly_features_transformed = self.poly.transform(df_for_poly)
        
        poly_feature_names = self.poly.get_feature_names_out(poly_cols)
        df_poly_with_all_terms = pd.DataFrame(poly_features_transformed, columns=poly_feature_names, index=current_df.index)
        
        # Birleştirme sırasında yinelenen sütunları önlemek için, df_poly_with_all_terms bunları yeniden 
        # ekleyeceği için orijinal temel sütunları (poly_cols) current_df'den bırakın.
        cols_to_drop_from_current = [col for col in poly_cols if col in current_df.columns]
        if cols_to_drop_from_current:
            current_df.drop(columns=cols_to_drop_from_current, inplace=True, errors='ignore')
        
        # Şimdi, current_df'i (artık orijinal 'Age', 'Fare' içermeyen) 
        # df_poly_with_all_terms ile birleştirin.
        current_df = pd.concat([current_df, df_poly_with_all_terms], axis=1)
        current_df.reset_index(drop=True, inplace=True) # concat sonrası index'i tekrar resetle

        # Adım 7: Etkileşim Özellikleri
        age_input_for_cut = None
        if 'Age' in current_df.columns:
            age_data = current_df['Age']
            if isinstance(age_data, pd.DataFrame):
                age_data = age_data.squeeze()
            if isinstance(age_data, pd.Series) and age_data.notna().any():
                age_input_for_cut = age_data
            else: 
                if hasattr(self.numeric_imputer, 'statistics_') and hasattr(self.numeric_imputer, 'feature_names_in_') and 'Age' in self.numeric_imputer.feature_names_in_:
                    age_default_value = self.numeric_imputer.statistics_[list(self.numeric_imputer.feature_names_in_).index('Age')]
                else:
                    age_default_value = 30 
                age_input_for_cut = pd.Series(np.full(len(current_df), age_default_value), index=current_df.index)
        else: 
            if hasattr(self.numeric_imputer, 'statistics_') and hasattr(self.numeric_imputer, 'feature_names_in_') and 'Age' in self.numeric_imputer.feature_names_in_:
                age_default_value = self.numeric_imputer.statistics_[list(self.numeric_imputer.feature_names_in_).index('Age')]
            else:
                age_default_value = 30
            age_input_for_cut = pd.Series(np.full(len(current_df), age_default_value), index=current_df.index)
            
        current_df['Age_Group'] = pd.cut(age_input_for_cut, 
                                 bins=[0, 12, 20, 40, 60, 100], labels=[0,1,2,3,4], include_lowest=True, right=False)
        current_df['Age_Group'] = current_df['Age_Group'].astype(int)

        def multiply_if_exists(df_in, col1, col2, new_col_name):
            # Çarpma öncesi Series'lerin index'lerinin aynı olduğundan emin olmaya gerek yok, pandas halleder ama tip önemli.
            if col1 in df_in.columns and col2 in df_in.columns:
                # Ensure operands are Series and numeric before multiplication
                op1 = df_in[col1]
                if isinstance(op1, pd.DataFrame):
                    op1 = op1.squeeze()
                
                op2 = df_in[col2]
                if isinstance(op2, pd.DataFrame):
                    op2 = op2.squeeze()

                s1 = pd.to_numeric(op1, errors='coerce').fillna(0)
                s2 = pd.to_numeric(op2, errors='coerce').fillna(0)
                df_in[new_col_name] = s1 * s2
            else:
                df_in[new_col_name] = 0 
            return df_in

        current_df = multiply_if_exists(current_df, 'Sex_Code', 'Pclass', 'Sex*Class')
        current_df = multiply_if_exists(current_df, 'Age_Group', 'Sex_Code', 'Age_Group*Sex')
        current_df = multiply_if_exists(current_df, 'Embarked_Code', 'Pclass', 'Embarked*Class')
        current_df = multiply_if_exists(current_df, 'FamilySize', 'Pclass', 'Family*Class')
        current_df = multiply_if_exists(current_df, 'Fare', 'Cabin_Location_Code', 'Fare*Cabin_Location_Code')
        current_df = multiply_if_exists(current_df, 'Age', 'FamilySize', 'Age*FamilySize')
        
        if 'Fare' in current_df.columns and 'FamilySize' in current_df.columns:
            current_df['FarePerPerson'] = (pd.to_numeric(current_df['Fare'], errors='coerce').fillna(0).replace(0, 1e-6) / 
                                       pd.to_numeric(current_df['FamilySize'], errors='coerce').fillna(1).replace(0, 1))
            current_df = multiply_if_exists(current_df, 'FarePerPerson', 'Pclass', 'FarePerPerson*Class')
        else:
            current_df['FarePerPerson'] = 0
            current_df['FarePerPerson*Class'] = 0

        current_df = multiply_if_exists(current_df, 'Cabin_Location_Code', 'Age_Group', 'Cabin_Location_Code*Age_Group')
        current_df = multiply_if_exists(current_df, 'Title_Code', 'Pclass', 'Title_Code*Class')
        
        if 'IsAlone' in current_df.columns and 'Age' in current_df.columns:
            current_df['Loneliness_Score'] = ((current_df['IsAlone'] == 1) & (pd.to_numeric(current_df['Age'], errors='coerce').fillna(0) > 30)).astype(int)
        else:
            current_df['Loneliness_Score'] = 0

        current_df = multiply_if_exists(current_df, 'TicketFrequency', 'Pclass', 'TicketFrequency*Class')
        current_df = multiply_if_exists(current_df, 'TicketPrefix_Code', 'Pclass', 'TicketPrefix_Code*Class')
        current_df = multiply_if_exists(current_df, 'Fare^2', 'Sex_Code', 'Fare^2*Sex_Code')
        current_df = multiply_if_exists(current_df, 'Age Fare', 'Sex_Code', 'Age Fare*Sex_Code')

        # Adım 8: Sütunları Düşürme
        columns_to_drop_base = ['Name', 'Ticket', 'Cabin', 'Sex', 'Cabin_Letter', 
                           'Cabin_Location', 'Embarked', 'Title', 'Age_Group',
                           'TicketPrefix', 'FarePerPerson' 
                          ]
        
        final_columns_to_drop = list(columns_to_drop_base) 

        if is_train:
            if 'PassengerId' in current_df.columns: # Eğitim setindeyse PassengerId'yi düşür
                final_columns_to_drop.append('PassengerId')
        # Test seti için PassengerId, final_columns_to_drop'a eklenmez. Aşağıda özel olarak ele alınacak.
        
        current_df.drop(columns=[col for col in final_columns_to_drop if col in current_df.columns], inplace=True, errors='ignore')
        
        # Test seti için PassengerId'nin doğru şekilde (başta) olduğundan emin ol
        if not is_train:
            if passenger_id_series_for_test is not None: # Orijinal PassengerId'ler en başta saklandıysa
                # current_df'de şu anda bir 'PassengerId' sütunu olabilir (eğer drop listesinde yoktuysa ve başta varsa)
                # veya olmayabilir (eğer orijinal df'de yoktu).
                # Her durumda, mevcut olanı (varsa) kaldırıp, sakladığımız orijinali ekleyelim ki doğru değerler ve index olsun.
                if 'PassengerId' in current_df.columns:
                    current_df.drop(columns=['PassengerId'], inplace=True, errors='ignore')
                current_df.insert(0, 'PassengerId', passenger_id_series_for_test.reset_index(drop=True))
            elif 'PassengerId' in current_df.columns: 
                # Orijinal ID'ler saklanmadı (örn: test_df'de yoktu), ama bir şekilde current_df'de 'PassengerId' var.
                # Varsa ve ilk sütun değilse başa taşı.
                if current_df.columns.get_loc('PassengerId') != 0:
                    pid_col_data = current_df.pop('PassengerId')
                    current_df.insert(0, 'PassengerId', pid_col_data)
            # else: passenger_id_series_for_test is None AND 'PassengerId' not in current_df.
            # Bu durumda test seti için PassengerId yok demektir. main() içindeki fallback bunu ele alabilir.

        return current_df

    def plot_feature_importance(self, model_to_plot, feature_names):
        """Özellik önem derecelerini görselleştir"""
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model_to_plot.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importances)
        plt.title('Özellik Önem Dereceleri')
        plt.tight_layout()
        plt.show()

    def plot_survival_by_feature(self, df, feature, title):
        """Belirli bir özelliğe göre hayatta kalma oranlarını görselleştir"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature, y='Survived', data=df, ci=None)
        plt.title(f'Hayatta Kalma Oranı - {title}')
        plt.ylabel('Hayatta Kalma Oranı')
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, df):
        """Korelasyon matrisini görselleştir"""
        # Grafik boyutunu daha büyük yapalım ve yazı tipi boyutunu ayarlayalım
        plt.figure(figsize=(20, 16)) # Boyutu artırdık (genişlik, yükseklik)
        sns.heatmap(df.corr(), 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0, 
                    fmt=".2f", # Ondalıkları 2 basamakla sınırla
                    linewidths=.5, # Hücreler arasına ince çizgiler ekle
                    annot_kws={"size": 8}) # Annotasyon (sayı) boyutunu küçülttük
        plt.title('Özellikler Arası Korelasyon Matrisi', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10) # X ekseni etiketlerini döndür ve hizala
        plt.yticks(fontsize=10) # Y ekseni etiket boyutu
        plt.tight_layout()
        plt.show()

    def plot_age_distribution(self, df):
        """Yaş dağılımını görselleştir"""
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=df, x='Age', hue='Survived', fill=True)
        plt.title('Yaşa Göre Hayatta Kalma Dağılımı')
        plt.xlabel('Yaş')
        plt.ylabel('Yoğunluk')
        plt.tight_layout()
        plt.show()

    def plot_fare_distribution(self, df):
        """Bilet ücreti dağılımını görselleştir"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=df)
        plt.title('Yolcu Sınıfı ve Bilet Ücretine Göre Hayatta Kalma')
        plt.xlabel('Yolcu Sınıfı')
        plt.ylabel('Bilet Ücreti')
        plt.tight_layout()
        plt.show()

    def plot_family_survival(self, df):
        """Aile büyüklüğüne göre hayatta kalma oranını görselleştir"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x='FamilySize', y='Survived', data=df, ci=None)
        plt.title('Aile Büyüklüğüne Göre Hayatta Kalma Oranı')
        plt.xlabel('Aile Büyüklüğü')
        plt.ylabel('Hayatta Kalma Oranı')
        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self, X_test, y_test):
        """Tüm modeller için ROC eğrilerini çiz"""
        plt.figure(figsize=(10, 6))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Rastgele Tahmin')
        plt.xlabel('Yanlış Pozitif Oranı')
        plt.ylabel('Doğru Pozitif Oranı')
        plt.title('ROC Eğrileri Karşılaştırması')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, X_test, y_test):
        """Tüm modeller için karmaşıklık matrislerini görselleştir"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self):
        """Model performanslarını karşılaştır"""
        accuracies = []
        model_names = []
        
        for name, model in self.models.items():
            accuracies.append(model.best_score_ * 100)
            model_names.append(name)
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(model_names))
        
        plt.bar(x, accuracies, color=['lightcoral', 'lightblue', 'lightgreen', 'lightgray', 'wheat'])
        plt.title('Model Doğruluk Oranları Karşılaştırması')
        plt.xlabel('Modeller')
        plt.ylabel('Doğruluk Oranı (%)')
        plt.xticks(x, model_names, rotation=45)
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.5, f'%{v:.1f}', ha='center')
        
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()

    def compare_best_model_predictions(self, X_test, y_test, y_pred):
        """En iyi model tahminlerini gerçek verilerle karşılaştır"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Test verisini DataFrame'e çevir
        test_df = self.data.iloc[y_test.index].copy()
        test_df['Tahmin'] = y_pred
        
        # 1. Cinsiyet Analizi
        ax = axes[0]
        sex_actual = test_df.groupby('Sex')['Survived'].mean() * 100
        sex_pred = test_df.groupby('Sex')['Tahmin'].mean() * 100
        
        x = np.arange(len(sex_actual))
        width = 0.35
        
        ax.bar(x - width/2, sex_actual, width, label='Gerçek', color='lightcoral')
        ax.bar(x + width/2, sex_pred, width, label='Tahmin', color='lightblue')
        
        # Yüzdeleri ekle
        for i, (v1, v2) in enumerate(zip(sex_actual, sex_pred)):
            ax.text(i - width/2, v1 + 1, f'%{v1:.1f}', ha='center')
            ax.text(i + width/2, v2 + 1, f'%{v2:.1f}', ha='center')
        
        ax.set_title('Cinsiyete Göre: Gerçek vs Tahmin')
        ax.set_xticks(x)
        ax.set_xticklabels(['Kadın', 'Erkek'])
        ax.legend()
        ax.set_ylim(0, 100)
        
        # 2. Yolcu Sınıfı Analizi
        ax = axes[1]
        class_actual = test_df.groupby('Pclass')['Survived'].mean() * 100
        class_pred = test_df.groupby('Pclass')['Tahmin'].mean() * 100
        
        x = np.arange(len(class_actual))
        
        ax.bar(x - width/2, class_actual, width, label='Gerçek', color='lightcoral')
        ax.bar(x + width/2, class_pred, width, label='Tahmin', color='lightblue')
        
        # Yüzdeleri ekle
        for i, (v1, v2) in enumerate(zip(class_actual, class_pred)):
            ax.text(i - width/2, v1 + 1, f'%{v1:.1f}', ha='center')
            ax.text(i + width/2, v2 + 1, f'%{v2:.1f}', ha='center')
        
        ax.set_title('Yolcu Sınıfına Göre: Gerçek vs Tahmin')
        ax.set_xticks(x)
        ax.set_xticklabels(['1. Sınıf', '2. Sınıf', '3. Sınıf'])
        ax.legend()
        ax.set_ylim(0, 100)
        
        # 3. Biniş Limanı Analizi
        ax = axes[2]
        port_actual = test_df.groupby('Embarked')['Survived'].mean() * 100
        port_pred = test_df.groupby('Embarked')['Tahmin'].mean() * 100
        
        x = np.arange(len(port_actual))
        
        ax.bar(x - width/2, port_actual, width, label='Gerçek', color='lightcoral')
        ax.bar(x + width/2, port_pred, width, label='Tahmin', color='lightblue')
        
        # Yüzdeleri ekle
        for i, (v1, v2) in enumerate(zip(port_actual, port_pred)):
            ax.text(i - width/2, v1 + 1, f'%{v1:.1f}', ha='center')
            ax.text(i + width/2, v2 + 1, f'%{v2:.1f}', ha='center')
        
        ax.set_title('Biniş Limanına Göre: Gerçek vs Tahmin')
        ax.set_xticks(x)
        ax.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'])
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def train_models(self, X_train, y_train):
        """Modelleri eğit"""
        # Temel modeller için GridSearchCV'de kullanılacak ortak metrikler
        scoring_metrics = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro'
        }
        refit_metric = 'f1_macro' # En iyi parametreleri f1_macro'ya göre seç

        # Logistic Regression
        lr_params = {'C': [0.1, 1.0, 10], 'solver': ['liblinear'], 'max_iter': [1000, 2000]}
        self.models['Logistic Regression'] = GridSearchCV(
            LogisticRegression(random_state=42),
            lr_params,
            cv=5, # CV sayısını artırabiliriz
            scoring=scoring_metrics,
            refit=refit_metric,
            n_jobs=-1
        )
        
        # Decision Tree
        dt_params = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        self.models['Decision Tree'] = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            dt_params,
            cv=5,
            scoring=scoring_metrics,
            refit=refit_metric,
            n_jobs=-1
        )
        
        # KNN
        knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
        self.models['KNN'] = GridSearchCV(
            KNeighborsClassifier(),
            knn_params,
            cv=5,
            scoring=scoring_metrics,
            refit=refit_metric,
            n_jobs=-1
        )
        
        # Random Forest
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.models['Random Forest'] = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=5,
            scoring=scoring_metrics,
            refit=refit_metric,
            n_jobs=-1
        )
        
        # XGBoost
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.7, 1.0]
        }
        self.models['XGBoost'] = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss'), # use_label_encoder kaldırıldı
            xgb_params,
            cv=5,
            scoring=scoring_metrics,
            refit=refit_metric,
            n_jobs=-1
        )
        
        # Temel modelleri eğit ve en iyilerini sakla
        trained_base_models_for_ensemble = []
        print("\nTemel modeller eğitiliyor...")
        for name, model_cv in self.models.items():
            print(f"\n{name} modeli eğitiliyor...")
            model_cv.fit(X_train, y_train)
            print(f"En iyi F1-macro skoru ({refit_metric}): {model_cv.best_score_:.4f}")
            print(f"En iyi parametreler: {model_cv.best_params_}")
            # Accuracy'i de raporlayalım
            best_index = model_cv.best_index_
            accuracy_at_best_f1 = model_cv.cv_results_['mean_test_accuracy'][best_index]
            print(f"Bu F1 skoru için Accuracy: {accuracy_at_best_f1:.4f}")
            trained_base_models_for_ensemble.append((name, model_cv.best_estimator_))

        self.base_models_for_stacking = trained_base_models_for_ensemble

        # Ensemble Modelleri
        print("\nEnsemble modelleri oluşturuluyor ve eğitiliyor...")

        # Voting Classifier (Hard ve Soft)
        # Sadece en iyi temel modellerden bazılarını seçebiliriz veya hepsini kullanabiliriz.
        # Örneğin, en iyi 3 veya 4 modeli alabiliriz.
        # Şimdilik, en iyi parametrelerle eğitilmiş tüm temel modelleri kullanalım.
        estimators_for_voting = [(name, model) for name, model in self.base_models_for_stacking]

        if estimators_for_voting:
            # Hard Voting
            voting_clf_hard = VotingClassifier(estimators=estimators_for_voting, voting='hard')
            print("\nHard Voting Classifier eğitiliyor...")
            # VotingClassifier için cross_val_score kullanalım, çünkü kendi içinde GridSearchCV yok.
            # Veya doğrudan fit edip sonra test seti üzerinde değerlendirebiliriz.
            # Şimdilik doğrudan fit edip, ana döngüde test performansına bakacağız.
            voting_clf_hard.fit(X_train, y_train)
            self.models['Voting Hard'] = voting_clf_hard # .best_estimator_ veya .best_score_ yok, doğrudan model

            # Soft Voting (eğer tüm modeller predict_proba'yı destekliyorsa)
            supports_proba = all(hasattr(model, 'predict_proba') for _, model in estimators_for_voting)
            if supports_proba:
                voting_clf_soft = VotingClassifier(estimators=estimators_for_voting, voting='soft')
                print("\nSoft Voting Classifier eğitiliyor...")
                voting_clf_soft.fit(X_train, y_train)
                self.models['Voting Soft'] = voting_clf_soft
            else:
                print("\nSoft Voting için tüm temel modeller 'predict_proba'yı desteklemiyor.")

            # Stacking Classifier
            # Meta-learner olarak Logistic Regression kullanalım (yaygın bir seçim)
            # Stacking için de tüm temel modelleri kullanalım.
            # KNN'i de dahil edelim. estimators_for_voting listesi KNN'i zaten içeriyor olmalı.
            meta_learner = LogisticRegression(solver='liblinear', random_state=42)
            
            # StackingClassifier için CV önemlidir. Default cv=5 kullanır.
            stacking_clf = StackingClassifier(estimators=estimators_for_voting, 
                                          final_estimator=meta_learner,
                                          cv=5, # İç CV
                                          n_jobs=-1)
            print("\nStacking Classifier eğitiliyor...")
            stacking_clf.fit(X_train, y_train)
            self.models['Stacking'] = stacking_clf
            # Stacking için CV skorlarını hesapla ve sakla
            print("Stacking Classifier için CV skorları hesaplanıyor...")
            try:
                f1_scores_stacking = cross_val_score(stacking_clf, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
                acc_scores_stacking = cross_val_score(stacking_clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
                self.ensemble_model_cv_scores['Stacking'] = {
                    'f1_macro': np.mean(f1_scores_stacking),
                    'accuracy': np.mean(acc_scores_stacking)
                }
                print(f"  Stacking CV F1-macro: {self.ensemble_model_cv_scores['Stacking']['f1_macro']:.4f}, CV Accuracy: {self.ensemble_model_cv_scores['Stacking']['accuracy']:.4f}")
            except Exception as e:
                print(f"Stacking için CV skorları hesaplanırken HATA: {e}")
                self.ensemble_model_cv_scores['Stacking'] = {'f1_macro': 0, 'accuracy': 0}

            # VotingClassifier'lar için de CV skorlarını hesaplayalım (eğer eklendilerse)
            if 'Voting Hard' in self.models:
                print("Voting Hard için CV skorları hesaplanıyor...")
                try:
                    f1_vh = cross_val_score(self.models['Voting Hard'], X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
                    acc_vh = cross_val_score(self.models['Voting Hard'], X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
                    self.ensemble_model_cv_scores['Voting Hard'] = {'f1_macro': np.mean(f1_vh), 'accuracy': np.mean(acc_vh)}
                    print(f"  Voting Hard CV F1-macro: {self.ensemble_model_cv_scores['Voting Hard']['f1_macro']:.4f}, CV Accuracy: {self.ensemble_model_cv_scores['Voting Hard']['accuracy']:.4f}")
                except Exception as e:
                    print(f"Voting Hard için CV skorları hesaplanırken HATA: {e}")
                    self.ensemble_model_cv_scores['Voting Hard'] = {'f1_macro': 0, 'accuracy': 0}
            
            if 'Voting Soft' in self.models:
                print("Voting Soft için CV skorları hesaplanıyor...")
                try:
                    f1_vs = cross_val_score(self.models['Voting Soft'], X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
                    acc_vs = cross_val_score(self.models['Voting Soft'], X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
                    self.ensemble_model_cv_scores['Voting Soft'] = {'f1_macro': np.mean(f1_vs), 'accuracy': np.mean(acc_vs)}
                    print(f"  Voting Soft CV F1-macro: {self.ensemble_model_cv_scores['Voting Soft']['f1_macro']:.4f}, CV Accuracy: {self.ensemble_model_cv_scores['Voting Soft']['accuracy']:.4f}")
                except Exception as e:
                    print(f"Voting Soft için CV skorları hesaplanırken HATA: {e}")
                    self.ensemble_model_cv_scores['Voting Soft'] = {'f1_macro': 0, 'accuracy': 0}
        else:
            print("\nEnsemble modelleri için eğitilmiş temel model bulunamadı.")

    def compare_predictions(self, X_test, y_test):
        """Model tahminlerini gerçek verilerle karşılaştır"""
        plt.figure(figsize=(10, 6))
        
        # Test verisindeki gerçek değerler
        actual_survival = y_test.mean() * 100
        
        # Her model için tahminler
        predictions = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            pred_survival = (y_pred == 1).mean() * 100
            predictions.append((name, pred_survival))
        
        # Grafik için veriler
        names = ['Gerçek'] + [p[0] for p in predictions]
        values = [actual_survival] + [p[1] for p in predictions]
        colors = ['lightcoral'] + ['lightgreen'] * len(predictions)
        
        # Çubuk grafik
        plt.bar(range(len(names)), values, color=colors)
        plt.title('Gerçek ve Tahmin Edilen Hayatta Kalma Oranları')
        plt.xticks(range(len(names)), names, rotation=45)
        plt.ylabel('Hayatta Kalma Yüzdesi (%)')
        
        # Yüzdeleri ekle
        for i, v in enumerate(values):
            plt.text(i, v + 1, f'%{v:.1f}', ha='center')
        
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()

    def plot_data_analysis(self):
        """Veri setindeki temel özelliklerin hayatta kalma ile ilişkisini görselleştirir."""
        if self.data is None:
            print("Hata: plot_data_analysis için veri yüklenmemiş (self.data is None).")
            return

        df_to_analyze = self.data.copy()

        if 'Survived' not in df_to_analyze.columns:
            print("Hata: Hedef değişken 'Survived' veri setinde bulunamadı.")
            return

        print("Temel özellik analiz grafikleri çiziliyor...")

        # Analiz edilecek özellikler ve onlara özel işlemler için bir yapı
        # Her tuple: (feature_name_for_title, column_to_plot, creation_function_if_needed, required_base_columns)
        # creation_function: temp_df alır, işlenmiş temp_df'i ve kullanılacak asıl sütun adını döndürür.
        
        features_config = [
            ('Cinsiyet', 'Sex', None, ['Sex']), 
            ('Yolcu Sınıfı', 'Pclass', None, ['Pclass']),
            ('Biniş Limanı', 'Embarked', None, ['Embarked']),
            ('Yaş Grubu', 'AgeGroup', 
             lambda df: (pd.cut(df['Age'], bins=[0, 12, 20, 30, 40, 50, 60, 100],
                                labels=['0-12', '13-20', '21-30', '31-40', '41-50', '51-60', '60+'], right=False), 'AgeGroup'),
             ['Age']),
            ('Bilet Ücreti Grubu', 'FareGroup', 
             lambda df: (pd.qcut(df['Fare'].fillna(df['Fare'].median()), q=5, 
                                labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek', 'Çok Yüksek'], duplicates='drop'), 'FareGroup'),
             ['Fare']),
            ('Aile Tipi', 'FamilyType', 
             lambda df: (self.create_family_features(df.copy())['FamilySize'].pipe(lambda s: pd.cut(s, bins=[0, 1, 4, float('inf')], labels=['Yalnız', 'Küçük Aile (2-4)', 'Büyük Aile (5+)'])), 'FamilyType'),
             ['SibSp', 'Parch']),
            ('Unvan (Gruplanmış)', 'Title_Grouped', 
             lambda df: (df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
                         .apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs', 'Master'] else 'Other'), 'Title_Grouped'),
             ['Name'])
        ]

        for title_prefix, plot_col_name_suggestion, creation_logic, req_cols in features_config:
            temp_df = df_to_analyze.copy()
            
            # Gerekli temel sütunlar var mı kontrol et
            if not all(col in temp_df.columns for col in req_cols):
                print(f"'{title_prefix}' analizi için gerekli sütunlar ({req_cols}) bulunamadı. Atlanıyor.")
                continue

            final_plot_column = plot_col_name_suggestion

            if creation_logic:
                try:
                    # creation_logic df'i doğrudan değiştirebilir veya yeni bir series/kolon adı döndürebilir
                    # Eğer Series dönerse, onu df'e atayacağız.
                    created_data, actual_col_name_from_creation = creation_logic(temp_df)
                    temp_df[actual_col_name_from_creation] = created_data
                    final_plot_column = actual_col_name_from_creation
                except Exception as e:
                    print(f"'{title_prefix}' için özellik oluşturulurken hata: {e}. Atlanıyor.")
                    continue
            
            if final_plot_column not in temp_df.columns:
                print(f"Uyarı: '{final_plot_column}' ('{title_prefix}' için) özelliği veri setinde bulunamadı/oluşturulamadı. Atlanıyor.")
                continue

            if temp_df[final_plot_column].isnull().any():
                if pd.api.types.is_categorical_dtype(temp_df[final_plot_column]) or temp_df[final_plot_column].dtype == 'object':
                    mode_val = temp_df[final_plot_column].mode()
                    temp_df[final_plot_column].fillna(mode_val[0] if not mode_val.empty else "Unknown", inplace=True)
                elif pd.api.types.is_numeric_dtype(temp_df[final_plot_column]): # Sayısal ise
                    median_val = temp_df[final_plot_column].median()
                    temp_df[final_plot_column].fillna(median_val if pd.notna(median_val) else 0, inplace=True)

            try:
                plt.figure(figsize=(10, 6))
                survival_rates = temp_df.groupby(final_plot_column)['Survived'].mean().sort_values(ascending=False)
                if survival_rates.empty:
                    print(f"'{title_prefix}' ({final_plot_column}) için hayatta kalma oranı hesaplanamadı. Atlanıyor.")
                    plt.close()
                    continue

                sns.barplot(x=survival_rates.index.astype(str), y=survival_rates.values, palette='viridis')
                plt.title(f'{title_prefix} Özelliğine Göre Hayatta Kalma Oranı', fontsize=14)
                plt.ylabel('Hayatta Kalma Oranı', fontsize=12)
                plt.xlabel(title_prefix, fontsize=12)
                plt.xticks(rotation=45, ha='right')
                
                for i, v in enumerate(survival_rates.values):
                    if pd.notna(v):
                         plt.text(i, v + 0.01, f"{v:.2%}", color='black', ha='center', va='bottom')
            
                plt.ylim(0, max(1.05, survival_rates.max() * 1.1 if not survival_rates.empty else 1.05) )
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"'{title_prefix}' ({final_plot_column}) için grafik çizdirilirken bir hata oluştu: {e}")
                plt.close()

    def plot_learning_curve(self, estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Verilen bir model için öğrenme eğrisini çizer.
        """
        if axes is None:
            _, axes = plt.subplots(1, 1, figsize=(10, 6))

        axes.set_title(title)
        if ylim is not None:
            axes.set_ylim(*ylim)
        axes.set_xlabel("Eğitim örnekleri")
        axes.set_ylabel("Skor (F1-macro)")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, 
                           cv=cv, 
                           n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           scoring='f1_macro', # F1 skoruna göre çiz
                           return_times=True)
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)

        # Eğitim skoru eğrisini çiz
        axes.grid(True, alpha=0.3)
        axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Eğitim skoru")
        axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Çapraz doğrulama skoru")
        axes.legend(loc="best")
        axes.set_facecolor('#f9f9f9') # Hafif bir arka plan rengi
        return plt

    def plot_validation_curve(self, estimator, title, X, y, param_name, param_range, axes=None, ylim=None, cv=None, n_jobs=-1):
        """
        Verilen bir model ve hiperparametre için doğrulama eğrisini çizer.
        """
        train_scores, test_scores = validation_curve(
            estimator, X, y, 
            param_name=param_name, 
            param_range=param_range, 
            cv=cv, 
            scoring="f1_macro", # F1 skoruna göre çiz
            n_jobs=n_jobs)
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if axes is None:
            _, axes = plt.subplots(1, 1, figsize=(10, 6))
        
        axes.set_title(title)
        axes.set_xlabel(param_name)
        axes.set_ylabel("Skor (F1-macro)")
        if ylim is not None:
            axes.set_ylim(*ylim)
        
        lw = 2
        axes.plot(param_range, train_scores_mean, label="Eğitim skoru",
                     color="darkorange", lw=lw)
        axes.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        axes.plot(param_range, test_scores_mean, label="Çapraz doğrulama skoru",
                     color="navy", lw=lw)
        axes.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        axes.legend(loc="best")
        axes.grid(True, alpha=0.3)
        axes.set_facecolor('#f9f9f9') # Hafif bir arka plan rengi
        return plt

    def plot_model_comparison(self):
        """Model performanslarını karşılaştır"""
        accuracies = []
        model_names = []
        
        for name, model in self.models.items():
            accuracies.append(model.best_score_ * 100)
            model_names.append(name)
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(model_names))
        
        plt.bar(x, accuracies, color=['lightcoral', 'lightblue', 'lightgreen', 'lightgray', 'wheat'])
        plt.title('Model Doğruluk Oranları Karşılaştırması')
        plt.xlabel('Modeller')
        plt.ylabel('Doğruluk Oranı (%)')
        plt.xticks(x, model_names, rotation=45)
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.5, f'%{v:.1f}', ha='center')
        
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()

"""## Ana Fonksiyonu Çalıştır"""

# HÜCRE 6: Ana fonksiyonu çalıştır
def main():
    print("--- main() fonksiyonu başladı ---") # Eklendi
    predictor = TitanicPredictor()
    
    # 1. Veri Yükleme
    print("\nEğitim verisi yükleniyor...")
    train_df = predictor.load_data('train (2).csv')
    if train_df is None:
        print("Eğitim verisi yüklenemedi. Program sonlandırılıyor.")
        return

    # Keşifsel Veri Analizi Grafikleri (Orijinal train_df üzerinde)
    # plot_data_analysis orijinal self.data'yı kullandığı için geçici olarak atayalım
    print("\nKeşifsel Veri Analizi grafikleri çiziliyor...")
    predictor.data = train_df.copy() # plot_data_analysis'in kullanması için
    predictor.plot_data_analysis()
    predictor.data = None # Tekrar None yapalım, karışıklık olmasın

    print("\nTest verisi yükleniyor...")
    test_df = predictor.load_data('test (2).csv')
    if test_df is None:
        print("Test verisi yüklenemedi. Program sonlandırılıyor.")
        return
    
    # PassengerId'yi test setinden alıp saklayalım (submission için)
    test_passenger_ids = test_df['PassengerId'].copy()

    # 2. Veri Ön İşleme
    print("\nEğitim verisi ön işleniyor...")
    y_train_full = train_df['Survived'].copy()
    train_df_features = train_df.drop('Survived', axis=1)
    
    processed_train_df = predictor.preprocess_data(train_df_features, is_train=True)

    # Korelasyon Matrisi (İşlenmiş eğitim verisi üzerinde)
    # Survived'ı da ekleyerek korelasyona bakalım
    temp_train_for_corr = processed_train_df.copy()
    temp_train_for_corr['Survived'] = y_train_full.values # Indexleri eşleşmeli
    print("\nÖn işlenmiş eğitim verisi için korelasyon matrisi çiziliyor...")
    predictor.plot_correlation_matrix(temp_train_for_corr)
    del temp_train_for_corr

    print("\nTest verisi ön işleniyor...")
    processed_test_df = predictor.preprocess_data(test_df.copy(), is_train=False)

    if 'PassengerId' not in processed_test_df.columns and 'PassengerId' in test_df.columns:
         processed_test_df['PassengerId'] = test_passenger_ids

    features_for_scaling = [col for col in processed_train_df.columns if col not in ['Survived', 'PassengerId']]
    X_train_processed = processed_train_df[features_for_scaling].copy()
    
    X_test_processed = pd.DataFrame(columns=X_train_processed.columns)
    for col in X_train_processed.columns:
        if col in processed_test_df.columns:
            X_test_processed[col] = processed_test_df[col]
        else:
            print(f"UYARI: Test setinde '{col}' sütunu bulunamadı, 0 ile dolduruluyor.")
            X_test_processed[col] = 0 
            
    X_test_processed = X_test_processed[X_train_processed.columns]

    # 4. Veri Ölçeklendirme
    print("\nVeriler ölçeklendiriliyor...")
    X_train_scaled = predictor.scaler.fit_transform(X_train_processed)
    X_test_scaled = predictor.scaler.transform(X_test_processed)

    # X_train_scaled'i DataFrame'e geri dönüştürelim (özellik isimleri için)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features_for_scaling)

    # 5. Model Eğitimi
    print("\nModeller eğitiliyor (tüm eğitim verisi üzerinde)...")
    predictor.train_models(X_train_scaled_df, y_train_full)

    best_overall_model_name = None
    best_overall_f1_score = 0  # F1 skoruna göre en iyiyi seçeceğiz
    best_overall_model_obj = None
    
    model_performance_results = []

    print("\nModel Performansları (Eğitim Seti CV Skorları ve Test Seti Tahminleri):")
    # X_test_scaled, test seti için ölçeklenmiş özellikler
    # y_train_full, eğitim etiketleri. Test etiketleri yok (Kaggle formatı)
    # Bu yüzden CV skorlarına ve test seti için yapılan tahminlerin submission'ına odaklanacağız.

    for name, model_obj in predictor.models.items():
        is_grid_search_cv = hasattr(model_obj, 'best_score_') and hasattr(model_obj, 'cv_results_')
        current_model_best_f1_cv = 0
        current_model_accuracy_cv = 0
        current_model_precision_cv = 0
        current_model_recall_cv = 0
        
        print(f"\n--- {name} ---")
        if is_grid_search_cv:
            best_idx = model_obj.best_index_
            current_model_best_f1_cv = model_obj.cv_results_['mean_test_f1_macro'][best_idx]
            current_model_accuracy_cv = model_obj.cv_results_['mean_test_accuracy'][best_idx]
            current_model_precision_cv = model_obj.cv_results_['mean_test_precision_macro'][best_idx]
            current_model_recall_cv = model_obj.cv_results_['mean_test_recall_macro'][best_idx]
            print(f"  CV F1-macro: {current_model_best_f1_cv:.4f}")
            print(f"  CV Accuracy: {current_model_accuracy_cv:.4f}")
            print(f"  CV Precision-macro: {current_model_precision_cv:.4f}")
            print(f"  CV Recall-macro: {current_model_recall_cv:.4f}")
            final_model_to_use = model_obj.best_estimator_
        else: # Voting veya Stacking gibi doğrudan eğitilmiş modeller
            # Bu modeller için CV skorları train_models içinde hesaplanmadı.
            # cross_val_score ile burada hesaplayabiliriz veya sadece eğitim setine fit edildikleri için CV'siz kabul edebiliriz.
            # Şimdilik CV skorlarını 0 olarak bırakıp, en iyi model seçiminde sadece GridSearchCV sonuçlarını önceliklendirebiliriz
            # ya da bu ensemble modelleri için de CV yapabiliriz. 
            # Basitlik adına, bu modellerin CV skorlarını burada göstermeyelim, sadece test tahminlerine odaklanalım.
            # Ancak en iyi model seçiminde dikkate alınmaları için, fit edilmiş hallerini kullanacağız.
            print(f"  Bu model ({name}) için CV skorları doğrudan GridSearchCV'den alınmıyor.")
            # Eğer test seti etiketleri olsaydı, burada test performansı hesaplanırdı.
            # Şimdilik, bu modellerin CV F1'ini 0 varsayıyoruz, ama yine de en iyi model olabilirler (test setinde).
            # Gerçek bir senaryoda bunlar için de CV yapılmalı.
            final_model_to_use = model_obj
            # Geçici olarak CV F1'i 0 kabul edelim, bu modellerin seçilmesi için
            # test seti (gender_submission) üzerinden bir değerlendirme daha mantıklı olurdu.
            # Şimdilik en iyi model seçimini CV F1'e göre yapıyoruz.

        # En iyi modeli F1 skoruna göre güncelle (GridSearchCV'den gelenler için)
        if is_grid_search_cv and current_model_best_f1_cv > best_overall_f1_score:
            best_overall_f1_score = current_model_best_f1_cv
            best_overall_model_name = name
            best_overall_model_obj = final_model_to_use
        
        model_performance_results.append({
            'Model': name,
            'CV_F1_macro': current_model_best_f1_cv,
            'CV_Accuracy': current_model_accuracy_cv,
            'CV_Precision': current_model_precision_cv,
            'CV_Recall': current_model_recall_cv,
            'model_object': final_model_to_use
        })

    # Eğer hiç GridSearchCV modeli yoksa veya hepsi 0 F1 verdiyse, ensemble modellerinden birini seçebiliriz.
    # Şimdilik, eğer best_overall_model_obj hala None ise (yani hiçbir GridSearchCV modeli F1>0 vermediyse)
    # ve ensemble modeller varsa, onlardan birini (örn: Stacking) varsayılan olarak atayalım.
    if best_overall_model_obj is None and 'Stacking' in predictor.models:
        print("\nHiçbir temel modelden anlamlı CV F1 skoru alınamadı. Stacking modeli en iyi olarak deneniyor.")
        best_overall_model_name = 'Stacking'
        best_overall_model_obj = predictor.models['Stacking']
        # Stacking modelinin F1'ini (eğer hesaplandıysa) al, yoksa 0 bırak.
        # Bu senaryo için, en iyi F1 skorunu yine de 0 olarak tutalım, çünkü CV ile gelmedi.
    elif best_overall_model_obj is None and 'Voting Soft' in predictor.models:
        print("\nStacking de yok, Voting Soft en iyi olarak deneniyor.")
        best_overall_model_name = 'Voting Soft'
        best_overall_model_obj = predictor.models['Voting Soft']
    elif best_overall_model_obj is None and 'Voting Hard' in predictor.models:
        print("\nVoting Soft da yok, Voting Hard en iyi olarak deneniyor.")
        best_overall_model_name = 'Voting Hard'
        best_overall_model_obj = predictor.models['Voting Hard']

    print(f"\nEn iyi CV F1-macro skoruna sahip model: {best_overall_model_name} (F1 Skor: {best_overall_f1_score:.4f})")
    predictor.best_model = best_overall_model_obj

    # Model Karşılaştırma Grafiği (CV F1 Skorlarına göre)
    # Sadece CV F1 skoru > 0 olanları ve GridSearchCV sonuçlarını çizdirelim.
    plot_df_perf = pd.DataFrame([res for res in model_performance_results if res['CV_F1_macro'] > 0])
    if not plot_df_perf.empty:
        plot_df_perf = plot_df_perf.sort_values('CV_F1_macro', ascending=False).reset_index(drop=True)

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='Model', y='CV_F1_macro', data=plot_df_perf, palette='viridis')
        plt.title('Modellerin Çapraz Doğrulama (CV) F1-macro Skorları')
        plt.ylabel('CV F1-macro Skoru')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., 
                     height + 0.005, 
                     f'{height:.3f}', 
                     ha='center', va='bottom')
        plt.ylim(0, max(1.0, plot_df_perf['CV_F1_macro'].max() * 1.1) if not plot_df_perf.empty else 1.0)
        plt.tight_layout()
        plt.show()
    else:
        print("Çizdirilecek yeterli model performans verisi (CV F1 > 0) bulunamadı.")

    # En İyi Modelin Özellik Önem Dereceleri
    if predictor.best_model and hasattr(predictor.best_model, 'feature_importances_'):
        print(f"\nEn iyi model ({best_overall_model_name}) için özellik önem dereceleri çiziliyor...")
        # Özellik isimleri ölçeklendirilmiş X_train_processed'den (veya X_train_scaled_df'den) alınabilir
        predictor.plot_feature_importance(predictor.best_model, X_train_scaled_df.columns)
    else:
        print(f"\nEn iyi model ({best_overall_model_name}) özellik önemini desteklemiyor veya bir model seçilmedi.")

    # Öğrenme ve Doğrulama Eğrileri (En İyi Model için)
    if predictor.best_model:
        print(f"\nEn iyi model ({best_overall_model_name}) için Öğrenme Eğrisi çiziliyor...")
        # X_train_scaled_df ve y_train_full kullanılacak
        # best_overall_model_obj, en iyi parametrelerle eğitilmiş veya doğrudan ensemble modeldir.
        # Learning curve için estimator'ın klonlanmamış olması sorun yaratabilir, bu yüzden clone() kullanmak iyi bir pratik.
        from sklearn.base import clone
        estimator_for_curves = clone(best_overall_model_obj)
        
        # Eğer best_overall_model_obj bir GridSearchCV ise, .best_estimator_'unu almalıyız.
        # Ancak yukarıdaki mantıkta best_overall_model_obj zaten .best_estimator_ veya doğrudan model oluyor.

        try:
            predictor.plot_learning_curve(estimator_for_curves, 
                                          f"Öğrenme Eğrisi ({best_overall_model_name})", 
                                          X_train_scaled_df, 
                                          y_train_full, 
                                          cv=5, # CV sayısını belirleyebiliriz, örn: 5
                                          n_jobs=-1)
            plt.show()
        except Exception as e:
            print(f"Öğrenme eğrisi çizilirken hata: {e}")

        # Doğrulama Eğrisi (Örnek olarak, eğer model Random Forest ise max_depth için)
        # Bu kısım modele özel ve seçilen parametreye göre ayarlanmalı.
        # Sadece GridSearchCV ile bulunan ve belirli parametreleri olan modeller için mantıklı.
        param_name_for_validation = None
        param_range_for_validation = None
        
        # En iyi modelin GridSearchCV'den gelip gelmediğini ve hangi tip olduğunu kontrol et
        original_model_type_name = best_overall_model_name
        if original_model_type_name == 'Random Forest':
            param_name_for_validation = 'max_depth'
            # GridSearchCV'de kullanılan aralığa benzer bir aralık seçelim
            # best_params_ değerini de içermeli
            best_params_rf = predictor.models['Random Forest'].best_params_ if 'Random Forest' in predictor.models and hasattr(predictor.models['Random Forest'], 'best_params_') else None
            if best_params_rf and 'max_depth' in best_params_rf:
                # Örnek bir aralık, best_params_'a göre ayarlanabilir.
                # rf_params = {'max_depth': [5, 7, 10], ...} idi.
                param_range_for_validation = [3, 5, 7, 10, 12, 15]
            else: # Default aralık
                param_range_for_validation = np.arange(3, 16, 2)
        elif original_model_type_name == 'XGBoost':
            param_name_for_validation = 'n_estimators'
            # xgb_params = {'n_estimators': [100, 200], ...} idi.
            param_range_for_validation = [50, 100, 150, 200, 250]
        elif original_model_type_name == 'Decision Tree':
            param_name_for_validation = 'max_depth'
            param_range_for_validation = [2, 3, 5, 7, 10, 12]
            
        if param_name_for_validation and param_range_for_validation is not None:
            # estimator_for_curves'ı tekrar clone edelim, çünkü learning_curve onu fit etmiş olabilir.
            estimator_for_validation = clone(best_overall_model_obj) 
            # Doğrulama eğrisi için modelin temel bir örneğini almamız gerekebilir, 
            # GridSearchCV objesinin kendisini değil, içindeki estimator'ı.
            # best_overall_model_obj zaten en iyi estimator olmalı.
            
            # Parametrenin estimator'da var olduğundan emin olalım
            if hasattr(estimator_for_validation, 'get_params') and param_name_for_validation in estimator_for_validation.get_params():
                print(f"\n{best_overall_model_name} için '{param_name_for_validation}' parametresine göre Doğrulama Eğrisi çiziliyor...")
                try:
                    predictor.plot_validation_curve(estimator_for_validation, 
                                                  f"Doğrulama Eğrisi ({best_overall_model_name} vs {param_name_for_validation})",
                                                  X_train_scaled_df, 
                                                  y_train_full, 
                                                  param_name=param_name_for_validation,
                                                  param_range=param_range_for_validation,
                                                  cv=5, 
                                                  n_jobs=-1)
                    plt.show()
                except Exception as e:
                    print(f"Doğrulama eğrisi çizilirken hata ({param_name_for_validation}): {e}")
            else:
                print(f"'{param_name_for_validation}' parametresi {best_overall_model_name} modelinde bulunamadı.")

    # 6. Test Verisi Üzerinde Tahmin Yapma
    if predictor.best_model:
        print(f"\nEn iyi model ({best_overall_model_name}) ile test verisi üzerinde tahmin yapılıyor...")
        test_predictions = predictor.best_model.predict(X_test_scaled)

        # 7. Submission Dosyası Oluşturma
        print("\nSubmission dosyası oluşturuluyor...")
        submission_df = pd.DataFrame({
            'PassengerId': test_passenger_ids,
            'Survived': test_predictions.astype(int)
        })
        submission_filepath = 'submission.csv'
        submission_df.to_csv(submission_filepath, index=False)
        print(f"Submission dosyası '{submission_filepath}' olarak kaydedildi.")
    else:
        print("HATA: Eğitilmiş bir model bulunamadı, tahmin yapılamıyor.")

    # SHAP ile Model Yorumlama (En İyi Model İçin)
    if predictor.best_model:
        print(f"\nEn iyi model ({best_overall_model_name}) için SHAP analizi yapılıyor...")
        try:
            # Ağaç tabanlı modeller için TreeExplainer daha verimlidir.
            # Modelin tipine göre uygun bir explainer seçilebilir.
            if isinstance(predictor.best_model, (RandomForestClassifier, XGBClassifier, DecisionTreeClassifier)):
                explainer = shap.TreeExplainer(predictor.best_model)
            else: 
                # Diğer modeller için (örn. LogisticRegression, KNN) KernelExplainer kullanılabilir
                # KernelExplainer tüm eğitim verisini (X_train_scaled_df) veya bir alt kümesini bekler.
                # Daha yavaş olabilir.
                print(f"Uyarı: {best_overall_model_name} için shap.TreeExplainer uygun değil. shap.KernelExplainer deneniyor...")
                # KernelExplainer'ın eğitim verisi medyanlarını (veya bir örneklem) alması önerilir.
                # shap.sample ile veya medyanları kullanarak bir arkaplan veri kümesi oluşturulabilir.
                # Basitlik için, eğitim verisinin bir kısmını veya tamamını kullanabiliriz, ancak büyük veri setlerinde yavaş olabilir.
                # Küçük veri setleri için tamamını kullanmak kabul edilebilir.
                if X_train_scaled_df.shape[0] > 100: # Örnek bir eşik
                    background_data = shap.sample(X_train_scaled_df, 100) # 100 örnek al
                else:
                    background_data = X_train_scaled_df
                explainer = shap.KernelExplainer(predictor.best_model.predict_proba, background_data)
            
            # Test setinin ölçeklenmiş halini (X_test_scaled) DataFrame'e dönüştür, özellik isimleriyle.
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features_for_scaling)
            
            shap_values = explainer.shap_values(X_test_scaled_df)
            
            print("SHAP özet grafiği çiziliyor...")
            # Eğer model ikili sınıflandırma yapıyorsa ve shap_values bir liste ise (genellikle [neg_class_shap, pos_class_shap])
            # pozitif sınıfın SHAP değerlerini kullanırız (genellikle index 1)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                 shap.summary_plot(shap_values[1], X_test_scaled_df, plot_type="bar", show=False)
                 plt.title(f'SHAP Feature Importance for {best_overall_model_name} (Positive Class)')
                 plt.show()
                 
                 shap.summary_plot(shap_values[1], X_test_scaled_df, show=False)
                 plt.title(f'SHAP Summary Plot for {best_overall_model_name} (Positive Class)')
                 plt.show()
            else:
                 # Tek bir array döndürüyorsa (örn: bazı regresyon veya TreeExplainer'ın bazı versiyonları)
                 shap.summary_plot(shap_values, X_test_scaled_df, plot_type="bar", show=False)
                 plt.title(f'SHAP Feature Importance for {best_overall_model_name}')
                 plt.show()

                 shap.summary_plot(shap_values, X_test_scaled_df, show=False)
                 plt.title(f'SHAP Summary Plot for {best_overall_model_name}')
                 plt.show()

            # SHAP bitti, şimdi SHAP'ın önemli bulduğu bazı özellikler için plot_data_analysis'i tekrar çağıralım
            # SHAP özet grafiği özellik önemine göre sıralanır.
            # X_test_scaled_df.columns, shap_values ile aynı sırada olmalı.
            if 'shap_values' in locals() and 'X_test_scaled_df' in locals():
                # Ortalama mutlak SHAP değerlerine göre özellikleri sırala
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
                else:
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                shap_feature_importance = pd.DataFrame({
                    'feature': X_test_scaled_df.columns,
                    'importance': mean_abs_shap
                }).sort_values('importance', ascending=False)

                top_n_shap_features = shap_feature_importance['feature'].tolist()[:5] # İlk 5 özelliği al
                print(f"\nSHAP tarafından en önemli bulunan ilk 5 özellik: {top_n_shap_features}")
                
                # Bu özelliklerden hangileri train_df'de var veya plot_data_analysis tarafından oluşturulabilir?
                # plot_data_analysis AgeGroup ve FareGroup'u kendisi oluşturabilir.
                # Diğerleri (Sex, Pclass, Embarked gibi) train_df'de olmalı.
                # Etkileşim/polinomik özellikler (örn: Age*Class) train_df'de doğrudan olmayabilir.
                
                # Şimdilik, train_df'de olan veya plot_data_analysis'in üretebileceği bilinen özelliklere odaklanalım.
                # SHAP listesinden, orijinal train_df'de olan veya AgeGroup/FareGroup olanları seçelim.
                drawable_shap_features = []
                known_drawable_by_plot_data_analysis = ['AgeGroup', 'FareGroup', 'Sex', 'Pclass', 'Embarked']
                
                for shap_feature_name in top_n_shap_features:
                    # SHAP feature name'leri preprocess sonrası isimlerdir (örn: Title_Code, Age Fare)
                    # plot_data_analysis ise orijinal train_df üzerindeki sütunlarla çalışır.
                    # Bu eşleştirmeyi yapmak karmaşık olabilir.
                    # Şimdilik, eğer SHAP en önemli özellikler arasında bilinen çizilebilirleri bulursa onları çizdirelim.
                    # Örneğin, eğer SHAP 'Pclass'ı önemli buluyorsa, çiz.
                    # Bu kısım daha da geliştirilebilir, SHAP isimlerini orijinal isimlere eşlemek için.
                    if shap_feature_name in known_drawable_by_plot_data_analysis:
                        drawable_shap_features.append(shap_feature_name)
                    elif 'Age' in shap_feature_name and 'AgeGroup' not in drawable_shap_features : # SHAP 'Age' veya 'Age*Fare' gibi bir şey bulduysa AgeGroup'u çiz
                        drawable_shap_features.append('AgeGroup')
                    elif 'Fare' in shap_feature_name and 'FareGroup' not in drawable_shap_features: # SHAP 'Fare' veya 'Fare^2' gibi bir şey bulduysa FareGroup'u çiz
                         drawable_shap_features.append('FareGroup')
                    elif 'Sex' in shap_feature_name and 'Sex' not in drawable_shap_features:
                        drawable_shap_features.append('Sex')
                    elif 'Pclass' in shap_feature_name and 'Pclass' not in drawable_shap_features:
                        drawable_shap_features.append('Pclass')
                    elif 'Embarked' in shap_feature_name and 'Embarked' not in drawable_shap_features:
                         drawable_shap_features.append('Embarked')
                
                # Tekrarları kaldır
                drawable_shap_features = list(dict.fromkeys(drawable_shap_features))

                if drawable_shap_features:
                    print(f"\nSHAP tarafından önemli bulunan ve çizilebilen özellikler için detaylı analizler:")
                    # plot_data_analysis, self.data üzerinde çalışır, bu yüzden train_df'i tekrar atamamız gerek.
                    predictor.data = train_df.copy() 
                    predictor.plot_data_analysis()
                    predictor.data = None # Tekrar None yapalım
                else:
                    print("\nSHAP tarafından önemli bulunan özellikler için doğrudan çizilebilecek temel özellik bulunamadı.")

        except Exception as e:
            print(f"SHAP analizi veya sonrası grafik çizimi sırasında bir hata oluştu: {e}")
    else:
        print("HATA: Eğitilmiş bir model bulunamadı, SHAP analizi yapılamıyor.")

    # Model, Scaler ve Özellik İsimlerini Kaydet
    if predictor.best_model and 'features_for_scaling' in locals() and features_for_scaling:
        print("\nEğitilmiş model, predictor durumu ve özellik listesi kaydediliyor...")
        try:
            joblib.dump(predictor, 'titanic_predictor_state.joblib')
            joblib.dump(features_for_scaling, 'final_feature_names.joblib')
            print("  predictor durumu 'titanic_predictor_state.joblib' olarak kaydedildi.")
            print("  Özellik listesi 'final_feature_names.joblib' olarak kaydedildi.")
        except Exception as e:
            print(f"  Model/Özellikler kaydedilirken HATA: {e}")
    else:
        print("\nModel, predictor durumu veya özellik listesi kaydedilemedi (gerekli bileşenler eksik).")

    # GUI Başlatma Kontrolleri (Artık burada GUI başlatılmayacak, sadece log bırakabiliriz)
    print("\nGUI Başlatma için gerekli değişkenlerin durumu (titanic_model.py):")
    print(f"  predictor.best_model is not None: {predictor.best_model is not None}")
    if predictor.best_model is not None:
        print(f"  En iyi model: {best_overall_model_name}")
    print(f"  'features_for_scaling' in locals(): {'features_for_scaling' in locals()}")
    if 'features_for_scaling' in locals() and features_for_scaling:
        print(f"  len(features_for_scaling): {len(features_for_scaling)}")
    else:
        print("  features_for_scaling tanımlı değil veya boş.")
    print(f"  train_df is not None: {train_df is not None}")
    if train_df is not None:
        print(f"  train_df.shape: {train_df.shape}")

    # GUI Başlatma KISMI KALDIRILDI
    # if predictor.best_model and 'features_for_scaling' in locals() and features_for_scaling and train_df is not None:
    #     print("\nPassenger Inspector GUI başlatılıyor...")
    #     launch_passenger_inspector_gui(predictor, train_df.copy(), features_for_scaling)
    # else:
    #     print("\nGUI başlatılamadı: Gerekli model, özellik listesi veya eğitim verisi bulunamadı.")

    print("--- main() fonksiyonu tamamlandı ---") 

if __name__ == "__main__":
    main() 
