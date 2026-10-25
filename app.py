
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# 1. KONFIGURASI DAN DESAIN
# ==================================================
st.set_page_config(
    page_title="NusaData | Analisis Data Sains",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
    color: #172b4d;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #101d35;
    border-right: 1px solid #263957;
}

[data-testid="stSidebar"] * {
    color: #e7edf7;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 5px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: #7db8ff;
}

.hero {
    background: linear-gradient(115deg, #101d35 0%, #174a7e 65%, #2675b8 100%);
    padding: 32px 34px;
    border-radius: 18px;
    margin-bottom: 28px;
    color: white;
    border: 1px solid #284c76;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin: 0 0 10px 0;
}

.hero p {
    color: #dce9f8;
    font-size: 15px;
    margin: 4px 0;
    line-height: 1.7;
}

.eyebrow {
    color: #8fc7ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.section-title {
    color: #172b4d;
    font-size: 23px;
    font-weight: 750;
    margin-top: 10px;
    margin-bottom: 5px;
}

.section-desc {
    color: #697b93;
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 18px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e3eaf3;
    box-shadow: 0 3px 12px rgba(22, 48, 80, 0.035);
}

div[data-testid="stMetricLabel"] {
    color: #667991;
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    color: #173d68;
    font-weight: 750;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #e1e8f1;
    border-radius: 15px;
    background: white;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    font-weight: 600;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #e1e8f1;
    border-radius: 10px;
}

hr {
    border-color: #e1e8f1;
}

.small-muted {
    color: #6b7b90;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 2. MEMBACA DATA
# ==================================================
NAMA_FILE = "data_MARVIN_normalisasi_siappakai.xlsx"

@st.cache_data
def baca_data():
    data = pd.read_excel(
        NAMA_FILE,
        sheet_name="Data_Panjang"
    )

    kolom = [
        "Provinsi",
        "Tahun",
        "Jumlah_Rumah_Sakit_Umum",
        "Kepadatan_Penduduk_per_km2"
    ]

    data = data[kolom].copy()

    for kolom_angka in [
        "Tahun",
        "Jumlah_Rumah_Sakit_Umum",
        "Kepadatan_Penduduk_per_km2"
    ]:
        data[kolom_angka] = pd.to_numeric(
            data[kolom_angka], errors="coerce"
        )

    data = data.dropna(subset=kolom)
    data = data[data["Provinsi"].notna()]
    return data


try:
    df = baca_data()
except Exception as e:
    st.error(
        "Data tidak berhasil dibaca. Periksa nama file Excel, "
        "nama sheet Data_Panjang, dan lokasi file."
    )
    st.exception(e)
    st.stop()

if df.empty:
    st.error("Tidak ada data valid untuk dianalisis.")
    st.stop()


# ==================================================
# 3. FUNGSI BANTU
# ==================================================
def judul_halaman(judul, deskripsi):
    st.markdown(
        f'<div class="section-title">{judul}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="section-desc">{deskripsi}</div>',
        unsafe_allow_html=True
    )


def format_angka(nilai, desimal=2):
    return f"{nilai:,.{desimal}f}"


# ==================================================
# 4. HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">DATA SCIENCE • INDONESIA</div>
    <h1>NusaData Analytics</h1>
    <p><b>Analisis Kepadatan Penduduk dan Rumah Sakit Umum</b></p>
    <p>
        Eksplorasi data provinsi Indonesia melalui visualisasi,
        pengelompokan K-Means, dan prediksi Linear Regression.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 5. NAVIGASI
# ==================================================
st.sidebar.markdown("## 📊 NusaData")
st.sidebar.caption("DASHBOARD ANALITIK")
st.sidebar.divider()

menu = st.sidebar.radio(
    "NAVIGASI UTAMA",
    [
        "Beranda",
        "Eksplorasi Data",
        "K-Means Clustering",
        "Linear Regression",
        "Kesimpulan"
    ],
    label_visibility="visible"
)

st.sidebar.divider()
st.sidebar.markdown("### Tentang Proyek")
st.sidebar.caption(
    "Analisis hubungan kepadatan penduduk dan jumlah "
    "rumah sakit umum berdasarkan data provinsi dan tahun."
)
st.sidebar.caption("Metode: K-Means & Linear Regression")
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span class="small-muted">PROYEK DATA SAINS</span>',
    unsafe_allow_html=True
)


# ==================================================
# 6. BERANDA
# ==================================================
if menu == "Beranda":

    judul_halaman(
        "Ringkasan Penelitian",
        "Gambaran umum data yang digunakan dalam analisis."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Observasi",
        f"{len(df):,}",
        help="Jumlah baris data provinsi dan tahun."
    )
    col2.metric(
        "Provinsi",
        f"{df['Provinsi'].nunique():,}"
    )
    col3.metric(
        "Tahun Pengamatan",
        f"{df['Tahun'].nunique():,}"
    )

    st.write("")
    kiri, kanan = st.columns([1.15, 0.85], gap="large")

    with kiri:
        with st.container(border=True):
            st.subheader("Tentang Analisis")
            st.write(
                "Penelitian ini mengeksplorasi hubungan antara "
                "kepadatan penduduk dan jumlah rumah sakit umum "
                "di tingkat provinsi."
            )
            st.write(
                "**Tujuan analisis**"
            )
            st.markdown("""
            - Mengetahui pola dan distribusi data.
            - Mengidentifikasi kelompok provinsi-tahun dengan K-Means.
            - Mengevaluasi kemampuan kepadatan penduduk dalam memprediksi jumlah RSU.
            """)

    with kanan:
        with st.container(border=True):
            st.subheader("Variabel Penelitian")
            st.markdown("""
            **01 — Provinsi**  
            Wilayah pengamatan.

            **02 — Tahun**  
            Tahun data tersedia.

            **03 — Jumlah RSU**  
            Jumlah rumah sakit umum.

            **04 — Kepadatan Penduduk**  
            Jumlah penduduk per km².
            """)

    with st.container(border=True):
        st.subheader("Data Penelitian")
        st.caption(
            "Gunakan tabel ini untuk memeriksa data yang dipakai "
            "sebelum menafsirkan hasil model."
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    with st.container(border=True):
        st.subheader("Statistik Deskriptif")

        deskripsi = df[
            [
                "Jumlah_Rumah_Sakit_Umum",
                "Kepadatan_Penduduk_per_km2"
            ]
        ].describe().T

        deskripsi.index = [
            "Jumlah Rumah Sakit Umum",
            "Kepadatan Penduduk per km²"
        ]

        st.dataframe(
            deskripsi.round(2),
            use_container_width=True
        )

        st.caption(
            "Statistik mencakup rata-rata, simpangan baku, "
            "nilai minimum, kuartil, dan maksimum."
        )


# ==================================================
# 7. EKSPLORASI DATA
# ==================================================
elif menu == "Eksplorasi Data":

    judul_halaman(
        "Eksplorasi Data",
        "Kenali pola, distribusi, dan perbedaan data melalui lima visualisasi."
    )

    korelasi = df[
        [
            "Kepadatan_Penduduk_per_km2",
            "Jumlah_Rumah_Sakit_Umum"
        ]
    ].corr().iloc[0, 1]

    with st.container(border=True):
        st.subheader("Hubungan Kepadatan dan Jumlah RSU")

        c1, c2 = st.columns([0.25, 0.75])

        with c1:
            st.metric("Korelasi Pearson", f"{korelasi:.3f}")

        with c2:
            if abs(korelasi) < 0.2:
                kekuatan = "sangat lemah"
            elif abs(korelasi) < 0.4:
                kekuatan = "lemah"
            elif abs(korelasi) < 0.6:
                kekuatan = "sedang"
            elif abs(korelasi) < 0.8:
                kekuatan = "kuat"
            else:
                kekuatan = "sangat kuat"

            if korelasi > 0:
                arah = "positif"
            elif korelasi < 0:
                arah = "negatif"
            else:
                arah = "tidak menunjukkan arah linear"

            st.info(
                f"Hubungan linear pada data ini bersifat {arah} "
                f"dengan kekuatan {kekuatan}. Korelasi tidak "
                "membuktikan hubungan sebab-akibat."
            )

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.scatter(
            df["Kepadatan_Penduduk_per_km2"],
            df["Jumlah_Rumah_Sakit_Umum"],
            alpha=0.7
        )
        ax.set_xlabel("Kepadatan Penduduk per km²")
        ax.set_ylabel("Jumlah Rumah Sakit Umum")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Setiap titik mewakili satu observasi provinsi-tahun."
        )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Provinsi dengan Rata-rata RSU Tertinggi")

            top10 = (
                df.groupby("Provinsi")["Jumlah_Rumah_Sakit_Umum"]
                .mean()
                .nlargest(10)
                .sort_values()
            )

            fig, ax = plt.subplots(figsize=(7, 4.8))
            top10.plot(kind="barh", ax=ax)
            ax.set_xlabel("Rata-rata Jumlah RSU")
            ax.set_ylabel("")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Membandingkan rata-rata jumlah RSU pada tahun "
                "yang tersedia untuk setiap provinsi."
            )

    with col2:
        with st.container(border=True):
            st.subheader("Rata-rata RSU per Tahun")

            tahunan = (
                df.groupby("Tahun")["Jumlah_Rumah_Sakit_Umum"]
                .mean()
            )

            fig, ax = plt.subplots(figsize=(7, 4.8))
            tahunan.plot(marker="o", ax=ax)
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Rata-rata Jumlah RSU")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Memperlihatkan perubahan rata-rata RSU "
                "pada tahun pengamatan."
            )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Distribusi Jumlah RSU")

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(
                df["Jumlah_Rumah_Sakit_Umum"],
                bins=15,
                edgecolor="white"
            )
            ax.set_xlabel("Jumlah RSU")
            ax.set_ylabel("Frekuensi")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Histogram menunjukkan sebaran jumlah RSU "
                "dan membantu mengenali nilai ekstrem."
            )

    with col2:
        with st.container(border=True):
            st.subheader("Peta Panas RSU")

            pivot = df.pivot_table(
                index="Provinsi",
                columns="Tahun",
                values="Jumlah_Rumah_Sakit_Umum",
                aggfunc="mean"
            )

            fig, ax = plt.subplots(figsize=(7, 5.5))
            gambar = ax.imshow(
                pivot,
                aspect="auto",
                cmap="Blues"
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Provinsi")
            fig.colorbar(gambar, ax=ax, label="Jumlah RSU")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Warna lebih gelap menunjukkan nilai RSU "
                "yang lebih tinggi pada skala grafik."
            )


# ==================================================
# 8. K-MEANS CLUSTERING
# ==================================================
elif menu == "K-Means Clustering":

    judul_halaman(
        "K-Means Clustering",
        "Kelompokkan observasi berdasarkan kemiripan kepadatan penduduk dan jumlah RSU."
    )

    with st.container(border=True):
        st.subheader("Pengaturan Model")

        st.write(
            "K-Means mengelompokkan data berdasarkan jarak "
            "antarobservasi. Fitur distandardisasi menggunakan "
            "Z-Score agar perbedaan skala tidak mendominasi."
        )

        k = st.selectbox(
            "Pilih jumlah cluster (K)",
            [2, 3, 4, 5],
            index=1
        )

    fitur = [
        "Kepadatan_Penduduk_per_km2",
        "Jumlah_Rumah_Sakit_Umum"
    ]

    X = df[fitur]
    Xz = (X - X.mean()) / X.std(ddof=0)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    hasil = df.copy()
    hasil["Cluster"] = model.fit_predict(Xz)

    jumlah = (
        hasil["Cluster"]
        .value_counts()
        .sort_index()
    )

    ringkasan = pd.DataFrame({
        "Cluster": hasil["Cluster"],
        "Kepadatan_ZScore": Xz.iloc[:, 0],
        "RSU_ZScore": Xz.iloc[:, 1]
    }).groupby("Cluster").mean().round(3)

    st.write("")

    st.subheader("Hasil Pengelompokan")

    metrik = st.columns(k)

    for i, cluster in enumerate(sorted(jumlah.index)):
        metrik[i].metric(
            f"Cluster {cluster}",
            f"{jumlah[cluster]} observasi"
        )

    col1, col2 = st.columns([0.8, 1.2], gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Jumlah Anggota")
            st.bar_chart(jumlah)
            st.caption(
                "Jumlah observasi pada masing-masing kelompok."
            )

    with col2:
        with st.container(border=True):
            st.subheader("Karakteristik Cluster")
            st.dataframe(
                ringkasan,
                use_container_width=True
            )
            st.caption(
                "Z-Score positif berarti rata-rata fitur cluster "
                "berada di atas rata-rata keseluruhan."
            )

    with st.container(border=True):
        st.subheader("Peta Persebaran Cluster")

        fig, ax = plt.subplots(figsize=(9, 5.5))

        for cluster in sorted(hasil["Cluster"].unique()):
            mask = hasil["Cluster"] == cluster
            ax.scatter(
                Xz.loc[mask].iloc[:, 0],
                Xz.loc[mask].iloc[:, 1],
                label=f"Cluster {cluster}",
                alpha=0.75
            )

        ax.axhline(0, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(0, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Kepadatan Penduduk (Z-Score)")
        ax.set_ylabel("Jumlah RSU (Z-Score)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Setiap titik mewakili satu observasi provinsi-tahun. "
            "Nomor cluster merupakan label, bukan urutan peringkat."
        )

    with st.container(border=True):
        st.subheader("Data Hasil Clustering")
        st.dataframe(
            hasil,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Unduh Hasil Clustering CSV",
            data=hasil.to_csv(index=False).encode("utf-8-sig"),
            file_name="hasil_clustering.csv",
            mime="text/csv"
        )


# ==================================================
# 9. LINEAR REGRESSION
# ==================================================
elif menu == "Linear Regression":

    judul_halaman(
        "Linear Regression",
        "Uji kemampuan kepadatan penduduk dalam memprediksi jumlah rumah sakit umum."
    )

    with st.container(border=True):
        st.subheader("Tentang Model")
        st.write(
            "Variabel X adalah kepadatan penduduk per km², "
            "sedangkan target Y adalah jumlah RSU. Data uji "
            "dipisahkan berdasarkan provinsi agar observasi "
            "provinsi yang sama tidak tersebar pada data latih "
            "dan data uji sekaligus."
        )

    X = df[["Kepadatan_Penduduk_per_km2"]]
    y = df["Jumlah_Rumah_Sakit_Umum"]

    pembagi = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        pembagi.split(X, y, groups=df["Provinsi"])
    )

    model = LinearRegression()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    prediksi = model.predict(X.iloc[test_idx])
    aktual = y.iloc[test_idx]

    mae = mean_absolute_error(aktual, prediksi)
    rmse = np.sqrt(mean_squared_error(aktual, prediksi))
    r2 = r2_score(aktual, prediksi)

    a, b, c = st.columns(3)
    a.metric("MAE", format_angka(mae))
    b.metric("RMSE", format_angka(rmse))
    c.metric("R²", format_angka(r2, 3))

    st.caption(
        "MAE dan RMSE lebih kecil umumnya lebih baik. "
        "R² mendekati 1 umumnya menunjukkan kecocokan yang lebih baik."
    )

    col1, col2 = st.columns([0.9, 1.1], gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Persamaan Regresi")

            st.latex(
                rf"\hat{{Y}} = {model.intercept_:.3f}"
                rf" + ({model.coef_[0]:.6f})X"
            )

            st.write(
                f"**Koefisien:** {model.coef_[0]:.6f}"
            )
            st.write(
                f"**Intercept:** {model.intercept_:.3f}"
            )

            st.info(
                "Koefisien menggambarkan perubahan prediksi Y "
                "untuk kenaikan satu unit X pada model ini. "
                "Ini bukan bukti hubungan sebab-akibat."
            )

    with col2:
        with st.container(border=True):
            st.subheader("Aktual vs Prediksi")

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.scatter(
                aktual,
                prediksi,
                alpha=0.75
            )

            batas_min = min(aktual.min(), prediksi.min())
            batas_max = max(aktual.max(), prediksi.max())

            ax.plot(
                [batas_min, batas_max],
                [batas_min, batas_max],
                linestyle="--"
            )

            ax.set_xlabel("Jumlah RSU Aktual")
            ax.set_ylabel("Jumlah RSU Prediksi")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Titik yang dekat dengan garis putus-putus "
                "menunjukkan prediksi yang dekat dengan nilai aktual."
            )

    with st.container(border=True):
        st.subheader("Interpretasi Evaluasi")

        st.markdown(f"""
        - **MAE = {mae:.2f}:** rata-rata selisih absolut antara nilai aktual dan prediksi, dalam satuan jumlah RSU.
        - **RMSE = {rmse:.2f}:** ukuran kesalahan prediksi yang memberi bobot lebih besar pada kesalahan besar.
        - **R² = {r2:.3f}:** ukuran kinerja model pada data uji dibandingkan dengan prediksi berdasarkan rata-rata target.
        """)

        if r2 < 0:
            st.warning(
                "R² negatif: pada data uji ini, model lebih buruk "
                "daripada menggunakan rata-rata target sebagai prediksi."
            )
        elif r2 < 0.3:
            st.info(
                "R² relatif rendah. Kepadatan penduduk saja "
                "memiliki kemampuan terbatas untuk memprediksi "
                "jumlah RSU pada data uji."
            )
        else:
            st.info(
                "Nilai R² perlu ditafsirkan bersama MAE, RMSE, "
                "dan karakteristik data uji."
            )

    with st.container(border=True):
        st.subheader("Perbandingan Aktual dan Prediksi")

        hasil_prediksi = pd.DataFrame({
            "Provinsi": df.iloc[test_idx]["Provinsi"].values,
            "Tahun": df.iloc[test_idx]["Tahun"].values,
            "Kepadatan Penduduk": X.iloc[test_idx, 0].values,
            "RSU Aktual": aktual.values,
            "RSU Prediksi": np.round(prediksi, 2)
        })

        st.dataframe(
            hasil_prediksi,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Unduh Hasil Prediksi CSV",
            data=hasil_prediksi.to_csv(index=False).encode("utf-8-sig"),
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )


# ==================================================
# 10. KESIMPULAN
# ==================================================
elif menu == "Kesimpulan":

    judul_halaman(
        "Kesimpulan Analisis",
        "Ringkasan interpretasi dan batasan dari analisis yang dilakukan."
    )

    korelasi = df[
        [
            "Kepadatan_Penduduk_per_km2",
            "Jumlah_Rumah_Sakit_Umum"
        ]
    ].corr().iloc[0, 1]

    with st.container(border=True):
        st.subheader("01 — Hubungan Antarvariabel")
        st.write(
            f"Korelasi Pearson pada data yang digunakan sebesar "
            f"**{korelasi:.3f}**. Nilai ini menggambarkan arah dan "
            "kekuatan hubungan linear, tetapi tidak membuktikan "
            "bahwa kepadatan penduduk menyebabkan perubahan jumlah RSU."
        )

    with st.container(border=True):
        st.subheader("02 — Pengelompokan K-Means")
        st.write(
            "K-Means membentuk kelompok berdasarkan kemiripan "
            "kepadatan penduduk dan jumlah RSU. Karakteristik setiap "
            "kelompok harus ditentukan dari rata-rata fitur, bukan "
            "hanya berdasarkan nomor cluster."
        )

    with st.container(border=True):
        st.subheader("03 — Kemampuan Prediksi")
        st.write(
            "Linear Regression menguji apakah kepadatan penduduk "
            "dapat digunakan untuk memprediksi jumlah RSU. "
            "Kemampuan model perlu dinilai dari MAE, RMSE, dan R² "
            "pada data uji."
        )

    with st.container(border=True):
        st.subheader("04 — Batasan Penelitian")
        st.write(
            "Analisis hanya menggunakan dua variabel utama dan "
            "tahun yang tersedia. Jumlah RSU tidak secara langsung "
            "menggambarkan kapasitas pelayanan, jumlah tempat tidur, "
            "akses masyarakat, atau distribusi fasilitas di dalam "
            "suatu provinsi. Faktor lain juga dapat memengaruhi "
            "persebaran rumah sakit."
        )

    st.info(
        "Rekomendasi: analisis lanjutan dapat mempertimbangkan "
        "variabel tambahan yang tersedia dan relevan, serta "
        "memeriksa kualitas dan konsistensi sumber data."
    )


# ==================================================
# 11. FOOTER
# ==================================================
st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:12px 0; color:#78879a;">
        <b>NusaData Analytics</b><br>
        Proyek Data Sains • Analisis Provinsi Indonesia<br>
        <span style="font-size:12px;">
        Hasil analisis merupakan ringkasan statistik dan model,
        bukan bukti sebab-akibat.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
