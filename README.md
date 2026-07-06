# Klasifikasi Kematangan Buah Kakao 🍫

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhrdianbaihaqi/cocoa-ripeness-classification/blob/main/notebooks/cocoa_ripeness_FINAL.ipynb)

Klasifikasi kesiapan panen buah kakao dari citra menggunakan **deep learning**
(EfficientNet-B0, transfer learning). Proyek ini menekankan pendekatan **data-centric**:
menganalisis dan memperbaiki kualitas data, bukan hanya menyetel model.

## Tim Penyusun

| Nama | NIM | Kelas |
|---|---|---|
| _Muhammad Rahardian Baihaqi_ | _1237050023_ | _Kelas ML-C_ |
| _Muhammad Ridwan Nur Ihsan_ | _1237050090_ | _Kelas ML-C_ |
| _Muhammad Daffa Naufal Putra_ | _1237050060_ | _Kelas ML-C_ |

## Ringkasan Hasil

| Model | Kelas | Akurasi (test) | Macro-F1 |
|---|---|---|---|
| Baseline | 3 | 0,650 | 0,517 |
| **Final (biner)** | **2** | **0,739** | **0,665** |

Model final mengklasifikasikan buah menjadi **`siap_panen`** vs **`belum_siap_panen`**.

> **Temuan utama:** faktor pembatas performa adalah kualitas anotasi dataset (banyak
> bounding box menangkap latar belakang, bukan buah), bukan arsitektur model — dibuktikan
> melalui audit visual sistematis.

## Dataset

Menggunakan dataset publik **Cocoa Pod Maturity Stages** dari Kaggle (format YOLO, 6 kelas).

🔗 **Sumber dataset:** https://www.kaggle.com/datasets/aidooben/cocoa-pod-maturity-stages-dataset

Dataset **tidak disertakan** dalam repo ini (sudah tersedia publik & berukuran besar).
Notebook mengunduhnya otomatis. Untuk unduh manual:

```bash
pip install kaggle
kaggle datasets download -d aidooben/cocoa-pod-maturity-stages-dataset -p data/
unzip data/cocoa-pod-maturity-stages-dataset.zip -d data/raw
```

> Hormati lisensi & sitasi dataset asli sesuai halaman Kaggle-nya.

## Cara Menjalankan (Google Colab)

1. Klik badge **Open in Colab** di atas (atau buka `notebooks/cocoa_ripeness_FINAL.ipynb`).
2. Aktifkan GPU: **Runtime -> Change runtime type -> T4 GPU**.
3. Jalankan sel berurutan dari atas ke bawah.

## Model Terlatih

Bobot model final (`best_model.pt`) tersedia di bagian **Releases** repo ini
(tidak di-commit ke git karena ukurannya).

## Struktur Repo

```
notebooks/   Notebook utama (pipeline + demo Gradio)
reports/     Confusion matrix & rangkuman metrik
data/        (kosong di git - diisi lewat skrip unduh)
```

## Metodologi Singkat

Konversi anotasi deteksi (YOLO) -> citra klasifikasi (crop per pod) -> pemetaan kelas ->
pembagian data **anti-kebocoran** (per gambar asal) -> fine-tuning 2 tahap EfficientNet-B0
-> evaluasi (macro-F1, confusion matrix) -> demo web (Gradio + Grad-CAM).


## Sitasi

Jika memakai proyek ini, mohon sitasi dataset asli (Kaggle) dan repo ini.

## Lisensi

Kode: lihat berkas [LICENSE](LICENSE). Dataset mengikuti lisensi sumber aslinya di Kaggle.
