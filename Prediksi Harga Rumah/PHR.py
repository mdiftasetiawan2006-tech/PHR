import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠", layout="centered")

FEATURES = [
    "luas_tanah", "luas_bangunan", "jumlah_kamar", "jumlah_kamar_mandi",
    "usia_bangunan", "jarak_pusat_kota_km", "kelas_lokasi"
]
LOCATION_VALUES = {
    "Pinggiran": 1,
    "Perumahan berkembang": 2,
    "Pusat kota": 3,
    "Kawasan premium": 4,
}

@st.cache_resource
def train_model():
    """Membuat data contoh dan melatih model. Data ini hanya simulasi, bukan data transaksi."""
    rng = np.random.default_rng(42)
    n = 5000
    land = rng.uniform(60, 600, n)
    building = rng.uniform(30, 350, n)
    rooms = rng.integers(1, 7, n)
    bathrooms = rng.integers(1, 5, n)
    age = rng.integers(0, 36, n)
    distance = rng.uniform(0.5, 30, n)
    location = rng.integers(1, 5, n)

    # Rumus simulasi dalam rupiah; noise membuat model belajar pola, bukan rumus pasti.
    multiplier = np.select(
        [location == 1, location == 2, location == 3, location == 4],
        [1.00, 1.18, 1.45, 1.85]
    )
    price = (
        (land * 2_000_000)
        + (building * 4_500_000)
        + (rooms * 18_000_000)
        + (bathrooms * 12_000_000)
        - (age * 3_000_000)
        - (distance * 2_000_000)
    ) * multiplier
    price = np.maximum(price + rng.normal(0, 45_000_000, n), 150_000_000)

    data = pd.DataFrame({
        "luas_tanah": land,
        "luas_bangunan": building,
        "jumlah_kamar": rooms,
        "jumlah_kamar_mandi": bathrooms,
        "usia_bangunan": age,
        "jarak_pusat_kota_km": distance,
        "kelas_lokasi": location,
    })
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], price, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=180, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return model, metrics

model, metrics = train_model()

st.title("🏠 Prediksi Harga Rumah")
st.write(
    "Masukkan karakteristik rumah untuk memperoleh estimasi harga dari model "
    "machine learning."
)
st.warning(
    "Mode demonstrasi: model dilatih menggunakan **data sintetis/simulasi**, "
    "bukan data transaksi rumah sebenarnya. Hasil tidak boleh digunakan sebagai "
    "dasar jual-beli, kredit, atau penilaian resmi."
)

with st.form("house_form"):
    st.subheader("Karakteristik rumah")
    col1, col2 = st.columns(2)
    with col1:
        land_area = st.number_input("Luas tanah (m²)", min_value=20.0, max_value=5000.0, value=120.0, step=5.0)
        building_area = st.number_input("Luas bangunan (m²)", min_value=15.0, max_value=3000.0, value=90.0, step=5.0)
        bedrooms = st.number_input("Jumlah kamar tidur", min_value=1, max_value=20, value=3, step=1)
        bathrooms = st.number_input("Jumlah kamar mandi", min_value=1, max_value=15, value=2, step=1)
    with col2:
        age = st.number_input("Usia bangunan (tahun)", min_value=0, max_value=150, value=5, step=1)
        distance = st.number_input("Jarak ke pusat kota (km)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
        location_name = st.selectbox("Kategori lokasi", list(LOCATION_VALUES.keys()))
    submitted = st.form_submit_button("Prediksi harga", type="primary", use_container_width=True)

if submitted:
    errors = []
    if building_area > land_area:
        errors.append("Luas bangunan tidak boleh lebih besar daripada luas tanah.")
    if bedrooms < 1 or bathrooms < 1:
        errors.append("Jumlah kamar harus minimal 1.")
    if errors:
        for error in errors:
            st.error(error)
    else:
        row = pd.DataFrame([{
            "luas_tanah": land_area,
            "luas_bangunan": building_area,
            "jumlah_kamar": bedrooms,
            "jumlah_kamar_mandi": bathrooms,
            "usia_bangunan": age,
            "jarak_pusat_kota_km": distance,
            "kelas_lokasi": LOCATION_VALUES[location_name],
        }], columns=FEATURES)
        estimate = float(model.predict(row)[0])
        st.success("Estimasi harga berhasil dihitung.")
        st.metric("Perkiraan harga rumah", f"Rp {estimate:,.0f}".replace(",", "."))
        st.caption("Angka ini merupakan keluaran model demo dan bukan appraisal profesional.")

with st.expander("Tentang model"):
    st.write("Algoritma: Random Forest Regressor dari scikit-learn.")
    st.write(f"MAE pengujian pada data simulasi: Rp {metrics['mae']:,.0f}".replace(",", "."))
    st.write(f"R² pengujian pada data simulasi: {metrics['r2']:.3f}")
    st.write("Metrik tersebut hanya mengukur kecocokan pada data simulasi, bukan akurasi pasar rumah nyata.")

st.divider()
st.caption("Untuk hasil yang realistis, latih ulang model menggunakan dataset transaksi rumah yang valid, bersih, dan sesuai wilayah target.")
