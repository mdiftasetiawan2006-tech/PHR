# Aplikasi Prediksi Harga Rumah

Aplikasi Streamlit dengan model `RandomForestRegressor`.

## Catatan penting tentang data
Versi ini langsung dapat dijalankan karena menggunakan **data sintetis (simulasi)** untuk demonstrasi alur machine learning. Model ini tidak menggambarkan harga pasar rumah yang sebenarnya. Jangan gunakan estimasi demo untuk keputusan jual-beli, pembiayaan, atau penilaian resmi.

Untuk penggunaan nyata, ganti fungsi `train_model()` pada `app.py` agar membaca dataset transaksi rumah yang sah dan relevan. Dataset idealnya berisi harga transaksi aktual, luas tanah/bangunan, kamar, usia bangunan, lokasi, dan fitur lain. Lakukan pembersihan data, encoding lokasi, evaluasi pada data uji, serta pemantauan model.

## Struktur proyek

```text
prediksi_harga_rumah/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Menjalankan di komputer

1. Pastikan Python 3.10 atau lebih baru terpasang.
2. Buka terminal pada folder proyek.
3. Buat virtual environment (disarankan):

   Windows:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Pasang pustaka:
   ```bash
   pip install -r requirements.txt
   ```
5. Jalankan:
   ```bash
   streamlit run app.py
   ```
6. Buka alamat lokal yang ditampilkan Streamlit, biasanya `http://localhost:8501`.

## Deploy gratis melalui Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Unggah `app.py`, `requirements.txt`, `README.md`, dan `.gitignore` ke repository.
3. Masuk ke https://share.streamlit.io/ dan login dengan GitHub.
4. Pilih **Create app**, repository, branch (biasanya `main`), dan file utama `app.py`.
5. Klik **Deploy**. Streamlit akan memasang pustaka dari `requirements.txt`.
6. Setelah proses selesai, aplikasi akan memiliki URL publik.

## Validasi input
Aplikasi membatasi rentang angka melalui komponen input Streamlit dan memeriksa agar luas bangunan tidak melebihi luas tanah. Pesan kesalahan ditampilkan sebelum prediksi.

## Daftar pustaka
- Streamlit: antarmuka web.
- scikit-learn: Random Forest dan evaluasi model.
- pandas: pengelolaan fitur.
- NumPy: pembuatan data simulasi.
