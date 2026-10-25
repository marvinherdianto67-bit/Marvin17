
import pandas as pd
import os

# Nama file Excel
file_excel = "data_MARVIN_normalisasi_siappakai.xlsx"

# Memeriksa apakah file Excel tersedia
if not os.path.exists(file_excel):
    print("ERROR: File Excel tidak ditemukan!")
    print("Pastikan file Excel berada di folder yang sama dengan program Python.")
else:
    print("File Excel ditemukan.")

    # Membaca file Excel
    df = pd.read_excel(
        file_excel,
        sheet_name="Data_Panjang",
        engine="openpyxl"
    )

    print("\n=== 5 BARIS PERTAMA DATA ===")
    print(df.head())

    print("\n=== NAMA KOLOM ===")
    print(df.columns.tolist())

    print("\n=== UKURAN DATA ===")
    print("Jumlah baris dan kolom:", df.shape)

    print("\n=== INFORMASI DATA ===")
    df.info()

print("\n=== DAFTAR SHEET EXCEL ===")

xls = pd.ExcelFile(file_excel, engine="openpyxl")
print(xls.sheet_names)


# Membaca hasil normalisasi Z-Score
df_zscore = pd.read_excel(
    file_excel,
    sheet_name="Normalisasi_ZScore",
    engine="openpyxl"
)

print("\n=== HASIL NORMALISASI Z-SCORE ===")
print(df_zscore.head())

print("\n=== UKURAN DATA Z-SCORE ===")
print(df_zscore.shape)

print("\n=== NAMA KOLOM Z-SCORE ===")
print(df_zscore.columns.tolist())


# Statistik deskriptif hasil normalisasi
print("\n=== STATISTIK DESKRIPTIF Z-SCORE ===")
print(
    df_zscore[
        ["RSU_ZScore", "Kepadatan_ZScore"]
    ].describe()
)

# Memeriksa nilai kosong
print("\n=== PEMERIKSAAN NILAI KOSONG ===")
print(
    df_zscore[
        ["RSU_ZScore", "Kepadatan_ZScore"]
    ].isnull().sum()
)

print("\n=== STATISTIK DESKRIPTIF Z-SCORE ===")
print(df_zscore[["RSU_ZScore", "Kepadatan_ZScore"]].describe())

print("\n=== PEMERIKSAAN NILAI KOSONG ===")
print(df_zscore[["RSU_ZScore", "Kepadatan_ZScore"]].isnull().sum())


import matplotlib.pyplot as plt

# Visualisasi hubungan dua variabel hasil normalisasi
plt.figure(figsize=(8, 5))

plt.scatter(
    df_zscore["Kepadatan_ZScore"],
    df_zscore["RSU_ZScore"]
)

plt.title("Hubungan Kepadatan Penduduk dan Jumlah Rumah Sakit Umum")
plt.xlabel("Kepadatan Penduduk (Z-Score)")
plt.ylabel("Jumlah Rumah Sakit Umum (Z-Score)")
plt.grid(True)

plt.tight_layout()
plt.show()

# Analisis korelasi
korelasi = df_zscore[
    ["Kepadatan_ZScore", "RSU_ZScore"]
].corr()

print("\n=== MATRIKS KORELASI ===")
print(korelasi)

print(
    "\nNilai korelasi:",
    korelasi.loc["Kepadatan_ZScore", "RSU_ZScore"]
)


# VISUALISASI 2: Bar chart 10 provinsi
# dengan rata-rata jumlah rumah sakit tertinggi

rata_rsu = (
    df.groupby("Provinsi")["Jumlah_Rumah_Sakit_Umum"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
rata_rsu.sort_values().plot(kind="barh")

plt.title("10 Provinsi dengan Rata-rata Jumlah Rumah Sakit Umum Tertinggi")
plt.xlabel("Rata-rata Jumlah Rumah Sakit Umum")
plt.ylabel("Provinsi")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# VISUALISASI 3: Grafik garis jumlah RSU berdasarkan tahun

rata_tahun = (
    df.groupby("Tahun")["Jumlah_Rumah_Sakit_Umum"]
    .mean()
)

plt.figure(figsize=(8, 5))
plt.plot(
    rata_tahun.index,
    rata_tahun.values,
    marker="o"
)

plt.title("Rata-rata Jumlah Rumah Sakit Umum Berdasarkan Tahun")
plt.xlabel("Tahun")
plt.ylabel("Rata-rata Jumlah Rumah Sakit Umum")
plt.xticks(rata_tahun.index)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# VISUALISASI 4: Histogram jumlah rumah sakit umum

plt.figure(figsize=(8, 5))

plt.hist(
    df["Jumlah_Rumah_Sakit_Umum"],
    bins=10,
    edgecolor="black"
)

plt.title("Distribusi Jumlah Rumah Sakit Umum")
plt.xlabel("Jumlah Rumah Sakit Umum")
plt.ylabel("Frekuensi")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# VISUALISASI 5: Heatmap jumlah RSU berdasarkan provinsi dan tahun

data_heatmap = df.pivot_table(
    index="Provinsi",
    columns="Tahun",
    values="Jumlah_Rumah_Sakit_Umum",
    aggfunc="mean"
)

plt.figure(figsize=(10, 12))

plt.imshow(
    data_heatmap,
    aspect="auto",
    cmap="YlOrRd"
)

plt.title("Heatmap Jumlah Rumah Sakit Umum per Provinsi dan Tahun")
plt.xlabel("Tahun")
plt.ylabel("Provinsi")

plt.xticks(
    range(len(data_heatmap.columns)),
    data_heatmap.columns
)

plt.yticks(
    range(len(data_heatmap.index)),
    data_heatmap.index,
    fontsize=7
)

plt.colorbar(label="Jumlah Rumah Sakit Umum")
plt.tight_layout()
plt.show()


# ==========================================
# TAHAP 4: PEMODELAN K-MEANS CLUSTERING
# ==========================================

from sklearn.cluster import KMeans

# Memilih dua variabel hasil normalisasi
X = df_zscore[[
    "Kepadatan_ZScore",
    "RSU_ZScore"
]]

# Membuat model K-Means dengan 3 kelompok
model_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Melatih model dan menentukan kelompok setiap data
df_zscore["Cluster"] = model_kmeans.fit_predict(X)

# Menampilkan hasil pengelompokan
print("\n=== HASIL K-MEANS CLUSTERING ===")
print(
    df_zscore[
        [
            "Provinsi",
            "Tahun",
            "Kepadatan_ZScore",
            "RSU_ZScore",
            "Cluster"
        ]
    ].head(10)
)

# Menampilkan jumlah data di setiap kelompok
print("\n=== JUMLAH DATA SETIAP CLUSTER ===")
print(df_zscore["Cluster"].value_counts().sort_index())

# Menampilkan karakteristik setiap cluster
print("\n=== KARAKTERISTIK SETIAP CLUSTER ===")

karakteristik_cluster = df_zscore.groupby("Cluster")[
    ["Kepadatan_ZScore", "RSU_ZScore"]
].mean()

print(karakteristik_cluster.round(3))

# ==========================================
# EVALUASI JUMLAH CLUSTER DENGAN METODE ELBOW
# ==========================================

inertia = []
jumlah_cluster = range(1, 11)

for k in jumlah_cluster:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X)
    inertia.append(model.inertia_)
    
print("\n=== NILAI INERTIA METODE ELBOW ===")
for k, nilai in zip(jumlah_cluster, inertia):
    print(f"K = {k}, Inertia = {nilai:.2f}")


plt.figure(figsize=(8, 5))
plt.plot(jumlah_cluster, inertia, marker="o")

plt.title("Metode Elbow untuk Menentukan Jumlah Cluster")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia")
plt.xticks(list(jumlah_cluster))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# EVALUASI CLUSTER DENGAN SILHOUETTE SCORE
# ==========================================

from sklearn.metrics import silhouette_score

print("\n=== EVALUASI SILHOUETTE SCORE ===")

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    label_cluster = model.fit_predict(X)
    skor = silhouette_score(X, label_cluster)

    print(f"K = {k}, Silhouette Score = {skor:.4f}")

# ==========================================
# RINGKASAN AKHIR HASIL K-MEANS
# ==========================================

print("\n=== JUMLAH ANGGOTA SETIAP CLUSTER ===")
print(df_zscore["Cluster"].value_counts().sort_index())

print("\n=== RATA-RATA SETIAP CLUSTER (Z-SCORE) ===")
print(
    df_zscore.groupby("Cluster")[
        ["Kepadatan_ZScore", "RSU_ZScore"]
    ].mean().round(3)
)

print("\n=== DATA PADA CLUSTER 2 ===")
print(
    df_zscore.loc[
        df_zscore["Cluster"] == 2,
        [
            "Provinsi",
            "Tahun",
            "Kepadatan_ZScore",
            "RSU_ZScore"
        ]
    ].to_string(index=False)
)
print("\n=== HASIL CLUSTERING K-MEANS ===")

print("\nJumlah data setiap cluster:")
print(df_zscore["Cluster"].value_counts().sort_index())

print("\nRata-rata setiap cluster:")
print(
    df_zscore.groupby("Cluster")[
        ["Kepadatan_ZScore", "RSU_ZScore"]
    ].mean().round(3)
)
# ==========================================
# TAHAP 5: LINEAR REGRESSION
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# Menentukan variabel input (X) dan target (y)
X_regresi = df[["Kepadatan_Penduduk_per_km2"]]
y_regresi = df["Jumlah_Rumah_Sakit_Umum"]

# Membagi data berdasarkan provinsi.
# Seluruh tahun dari satu provinsi tetap berada
# pada kelompok train atau test yang sama.
pembagi = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

indeks_train, indeks_test = next(
    pembagi.split(X_regresi, y_regresi, groups=df["Provinsi"])
)

X_train = X_regresi.iloc[indeks_train]
X_test = X_regresi.iloc[indeks_test]
y_train = y_regresi.iloc[indeks_train]
y_test = y_regresi.iloc[indeks_test]

# Membuat dan melatih model
model_regresi = LinearRegression()
model_regresi.fit(X_train, y_train)

# Membuat prediksi pada data test
y_prediksi = model_regresi.predict(X_test)

# Evaluasi model
mae = mean_absolute_error(y_test, y_prediksi)
rmse = np.sqrt(mean_squared_error(y_test, y_prediksi))
r2 = r2_score(y_test, y_prediksi)

print("\n=== HASIL LINEAR REGRESSION ===")
print("Jumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))

print("\nKoefisien regresi:", model_regresi.coef_[0])
print("Intercept:", model_regresi.intercept_)

print("\n=== EVALUASI MODEL ===")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R²  :", round(r2, 3))

# Membandingkan nilai aktual dan prediksi
hasil_prediksi = pd.DataFrame({
    "Kepadatan_Penduduk": X_test[
        "Kepadatan_Penduduk_per_km2"
    ].values,
    "RSU_Aktual": y_test.values,
    "RSU_Prediksi": y_prediksi.round(2)
})

print("\n=== PERBANDINGAN AKTUAL DAN PREDIKSI ===")
print(hasil_prediksi.head(10).to_string(index=False))

# Visualisasi nilai aktual dan prediksi
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_prediksi)

batas_min = min(y_test.min(), y_prediksi.min())
batas_max = max(y_test.max(), y_prediksi.max())
plt.plot(
    [batas_min, batas_max],
    [batas_min, batas_max],
    linestyle="--"
)

plt.title("Linear Regression: Aktual vs Prediksi")
plt.xlabel("Jumlah Rumah Sakit Aktual")
plt.ylabel("Jumlah Rumah Sakit Prediksi")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
