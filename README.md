# Klasifikasi Kematangan Buah Kakao 🍫

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhrdianbaihaqi/cocoa-ripeness-classification/blob/main/notebooks/cocoa_ripeness_FINAL.ipynb)
[![Live Demo](https://img.shields.io/badge/Gradio-Live%20Demo-orange?logo=gradio)](https://43ee0ac70ce52119aa.gradio.live/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Aplikasi **deep learning** untuk mengklasifikasikan **kesiapan panen buah kakao**
langsung dari citra/foto buah. Proyek ini menekankan pendekatan **data-centric**:
performa ditingkatkan dengan menganalisis dan memperbaiki kualitas data, bukan sekadar
menyetel arsitektur model.

---

## 🚀 Demo Langsung

Coba modelnya secara interaktif tanpa perlu instalasi apa pun:

**🔗 https://43ee0ac70ce52119aa.gradio.live/**

Unggah foto buah kakao, lalu aplikasi akan memprediksi apakah buah **`siap_panen`**
atau **`belum_siap_panen`** lengkap dengan visualisasi **Grad-CAM** (menyorot bagian
gambar yang menjadi dasar keputusan model).

> ⚠️ **Catatan:** tautan `*.gradio.live` bersifat **sementara** (aktif ±72 jam selama
> sesi notebook berjalan). Jika tidak dapat diakses, jalankan ulang sel demo Gradio pada
> notebook untuk memperoleh tautan baru.

---

## 📌 Tentang Proyek

Petani kakao sering kesulitan menentukan waktu panen yang tepat. Buah yang dipanen
terlalu dini atau terlalu matang menurunkan kualitas dan nilai jual biji kakao.
Proyek ini membangun model klasifikasi citra yang membantu **mengotomatiskan penilaian
kematangan buah kakao** sehingga keputusan panen menjadi lebih cepat, konsisten, dan
objektif.

**Tujuan utama:**

- Membangun pipeline klasifikasi citra dari dataset deteksi (YOLO) publik.
- Menerapkan **transfer learning** (EfficientNet-B0) untuk klasifikasi kematangan.
- Mendemonstrasikan bahwa **kualitas data** adalah faktor pembatas utama performa,
  melalui audit anotasi yang sistematis.
- Menyediakan **demo web** yang mudah diakses (Gradio + Grad-CAM).

**Fitur:**

- ✅ Klasifikasi biner: `siap_panen` vs `belum_siap_panen`
- ✅ Transfer learning EfficientNet-B0 (fine-tuning 2 tahap)
- ✅ Pembagian data **anti-kebocoran** (split per gambar asal, bukan per crop)
- ✅ Evaluasi lengkap: akurasi, macro-F1, confusion matrix
- ✅ Explainability via **Grad-CAM**
- ✅ Antarmuka web interaktif (Gradio)

---

## 📊 Ringkasan Hasil

| Model | Kelas | Akurasi (test) | Macro-F1 |
|---|---|---|---|
| Baseline | 3 | 0,650 | 0,517 |
| **Final (biner)** | **2** | **0,739** | **0,665** |

Model final mengklasifikasikan buah menjadi **`siap_panen`** vs **`belum_siap_panen`**.

> **Temuan utama:** faktor pembatas performa adalah kualitas anotasi dataset (banyak
> bounding box menangkap latar belakang, bukan buah), bukan arsitektur model — dibuktikan
> melalui audit visual sistematis.

---

## 🗂️ Dataset

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

---

## ▶️ Cara Menjalankan

### Opsi 1 — Google Colab (direkomendasikan)

1. Klik badge **Open in Colab** di atas (atau buka `notebooks/cocoa_ripeness_FINAL.ipynb`).
2. Aktifkan GPU: **Runtime → Change runtime type → T4 GPU**.
3. Jalankan sel berurutan dari atas ke bawah.
4. Sel terakhir menghasilkan tautan **Gradio** publik untuk demo.

### Opsi 2 — Lokal

```bash
# 1. Kloning repo
git clone https://github.com/rhrdianbaihaqi/cocoa-ripeness-classification.git
cd cocoa-ripeness-classification

# 2. Buat environment & pasang dependency
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Jalankan notebook
jupyter notebook notebooks/cocoa_ripeness_FINAL.ipynb
```

---

## 📁 Struktur Folder

```
cocoa-ripeness-classification/
├── notebooks/
│   └── cocoa_ripeness_FINAL.ipynb   # Notebook utama: pipeline lengkap + demo Gradio
├── reports/
│   ├── confusion_matrix.png         # Visualisasi confusion matrix model final
│   └── Rangkuman_Metrik.docx        # Rangkuman metrik & analisis hasil
├── data/
│   └── .gitkeep                     # Placeholder — data diunduh via skrip (lihat Dataset)
├── requirements.txt                 # Daftar dependency Python
├── LICENSE                          # Lisensi kode (MIT)
├── .gitignore                       # Berkas/artefak yang tidak di-commit
└── README.md                        # Dokumen ini
```

> Folder `data/`, bobot model (`*.pt`), dan artefak besar lain sengaja diabaikan git
> (lihat `.gitignore`). Bobot model final (`best_model.pt`) tersedia di bagian **Releases**.

---

## 🧪 Metodologi Singkat

```
Anotasi deteksi (YOLO)
        │  konversi
        ▼
Citra klasifikasi (crop per pod)
        │  pemetaan kelas → biner
        ▼
Split anti-kebocoran (per gambar asal)
        │
        ▼
Fine-tuning 2 tahap EfficientNet-B0 (transfer learning)
        │
        ▼
Evaluasi (macro-F1, confusion matrix)
        │
        ▼
Demo web (Gradio + Grad-CAM)
```

---

## 👥 Tim Penyusun

| Nama | NIM | Kelas |
|---|---|---|
| Muhammad Rahardian Baihaqi | 1237050023 | ML-C |
| Muhammad Ridwan Nur Ihsan | 1237050090 | ML-C |
| Muhammad Daffa Naufal Putra | 1237050060 | ML-C |

---

## 📚 Sitasi

Jika memakai proyek ini, mohon sitasi dataset asli (Kaggle) dan repo ini.

## 📄 Lisensi

Kode: lihat berkas [LICENSE](LICENSE). Dataset mengikuti lisensi sumber aslinya di Kaggle.
