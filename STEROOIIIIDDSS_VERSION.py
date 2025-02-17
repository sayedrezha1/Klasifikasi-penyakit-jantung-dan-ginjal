import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd

# Lazy loading using Dask for large datasets
def load_dataset_lazy(file_path):
    try:
        # Dask reads data lazily (in chunks) to avoid memory overload
        data = dd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File '{file_path}' tidak ditemukan. Periksa path dan lokasi file!")

# Lazy loading for the datasets
data_jantung = load_dataset_lazy('dataset/Dataset_Jantung.csv')
data_ginjal = load_dataset_lazy('dataset/Dataset_Ginjal.csv')

# Menampilkan web
st.title('Klasifikasi Penyakit Jantung dan Ginjal')
st.write("""
    # Menggunakan beberapa algoritma dan dataset yang berbeda
    #### Mana yang Terbaik?
    """
)

# Menampilkan pilihan dataset
nama_dataset = st.sidebar.selectbox(
    'Pilih Dataset',
    ('Penyakit Jantung', 'Penyakit Ginjal')
)

st.write(f"## Dataset {nama_dataset}")

# menampilkan pilihan algoritma
algoritma = st.sidebar.selectbox(
    'Pilih Algoritma',
    ('KNN', 'SVM', 'Random Forest')
)

# inisialisasi dataset yang akan di tampilkan
def pilih_dataset(nama):
    if nama == 'Penyakit Jantung':
        return data_jantung
    else:
        return data_ginjal

# Function to preprocess dataset lazily
def preprocess_dataset_lazily(dataset):
    # We can also optimize column types for memory
    dataset = dataset.astype({'age': 'float32', 'sex': 'category', 'cp': 'category'})
    # Preprocessing logic remains similar but applied lazily
    # Filling missing values using Dask lazy operations
    dataset = dataset.fillna(dataset.mean())
    return dataset

# Preprocessing dataset lazily
dataset = pilih_dataset(nama_dataset)

if dataset is None:
    st.error("Dataset tidak ditemukan atau gagal dimuat. Periksa pilihan dataset Anda.")
else:
    # Preprocessing dataset lazily
    dataset = preprocess_dataset_lazily(dataset)

    # Convert Dask DataFrame to Pandas for model training
    dataset = dataset.compute()  # Convert to pandas dataframe to fit the model

    # Pisahkan fitur (x) dan target (y)
    x = dataset.iloc[:, :-1]  # Semua kolom kecuali kolom terakhir
    y = dataset.iloc[:, -1]   # Kolom terakhir

    # Feature Scaling (for KNN and SVM)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Tampilkan informasi dataset
    st.write('Jumlah Baris dan Kolom : ', x.shape)
    st.write('Jumlah Kelas : ', y.nunique())

def tambah_parameter(nama_algoritma):
    params = dict()
    if nama_algoritma == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif nama_algoritma == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 1, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = tambah_parameter(algoritma)

def pilih_klasifikasi(nama_algoritma, params):
    if nama_algoritma == 'KNN':
        return KNeighborsClassifier(n_neighbors=params['K'], n_jobs=-1)  # Parallel KNN
    elif nama_algoritma == 'SVM':
        return SVC(C=params['C'], parallel_jobs=-1)  # Parallel SVM (may not fully utilize parallelism as SVM itself doesn't support multi-threading in scikit-learn)
    else:
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=12345,
            n_jobs=-1  # Enable multi-threading in Random Forest
        )

algorithm = pilih_klasifikasi(algoritma, params)

# Menampilkan hasil prediksi
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=12345)

# Parallelizing cross-validation with multiple threads
cross_val_score_result = cross_val_score(algorithm, x_train, y_train, cv=5, n_jobs=-1)

# Train and predict
algorithm.fit(x_train, y_train)
y_pred = algorithm.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f'Algoritma : {algoritma}')
st.write(f'Akurasi : {accuracy}')
st.write(f'Cross-Validation Accuracy: {cross_val_score_result.mean()}')

# Confusion Matrix and Classification Report
st.write('Confusion Matrix:')
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

st.write('Classification Report:')
st.text(classification_report(y_test, y_pred))

#  PLOT DATASET
pca = PCA(2)
x_projected = pca.fit_transform(x_scaled)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)
