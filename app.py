import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os

# ── Konfigurasi Halaman ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Alzheimer - CNN OASIS-1",
    page_icon="🧠",
    layout="wide"
)

# ── Konstanta ─────────────────────────────────────────────────────────────
MODEL_PATH = "model.tflite"
TARGET_SIZE = (224, 224)

# Metrik dari hasil evaluasi test set
METRICS = {
    "Akurasi": 0.959356,
    "Presisi": 0.858711,
    "Recall": 0.978125,
    "F1-Score": 0.914536,
    "AUC-ROC": 0.994393,
    "MCC": 0.891261,
}


# ── Load TFLite Model (cached) ───────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


# ── Preprocessing Citra ──────────────────────────────────────────────────
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, :, :, np.newaxis]  # (1, 224, 224, 1)


# ── Prediksi dengan TFLite ───────────────────────────────────────────────
def predict(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prob_ad = float(interpreter.get_tensor(output_details[0]["index"])[0][0])
    return prob_ad


# ── Sidebar Navigasi ─────────────────────────────────────────────────────
st.sidebar.title("🧠 Navigasi")
page = st.sidebar.radio(
    "Pilih Modul:",
    ["Prediksi Diagnosis", "Dashboard Evaluasi Model"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Skripsi:** Pengembangan dan Evaluasi Model CNN "
    "untuk Diagnosis Dini Penyakit Alzheimer "
    "Menggunakan Citra MRI OASIS-1"
)


# ══════════════════════════════════════════════════════════════════════════
# MODUL 1: PREDIKSI DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════
if page == "Prediksi Diagnosis":
    st.title("🧠 Prediksi Diagnosis Alzheimer")
    st.markdown(
        "Unggah citra MRI otak untuk memperoleh hasil klasifikasi "
        "**CN (Cognitively Normal)** atau **AD (Alzheimer's Disease)**."
    )

    uploaded = st.file_uploader(
        "Pilih file citra MRI",
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG",
    )

    if uploaded is not None:
        col1, col2 = st.columns(2)

        # ── Pratinjau Citra ───────────────────────────────────────────
        with col1:
            st.subheader("Citra MRI yang Diunggah")
            st.image(uploaded, use_container_width=True)

        # ── Hasil Prediksi ────────────────────────────────────────────
        with col2:
            st.subheader("Hasil Prediksi")

            with st.spinner("Memproses citra..."):
                interpreter = load_model()
                img_array = preprocess_image(uploaded)
                prob_ad = predict(interpreter, img_array)
                prob_cn = 1.0 - prob_ad
                is_ad = prob_ad >= 0.5

            # Label hasil
            if is_ad:
                st.error("🔴 **AD (Alzheimer's Disease)**")
            else:
                st.success("🟢 **CN (Cognitively Normal)**")

            # Confidence score
            confidence = prob_ad if is_ad else prob_cn
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")

            # Distribusi probabilitas (bar chart)
            st.markdown("**Distribusi Probabilitas:**")
            prob_df = pd.DataFrame(
                {
                    "Kelas": ["CN (Normal)", "AD (Alzheimer)"],
                    "Probabilitas": [prob_cn, prob_ad],
                }
            )
            st.bar_chart(prob_df.set_index("Kelas"), height=200)

            # Progress bar visual
            st.markdown("**Probabilitas AD:**")
            st.progress(prob_ad)

        # ── Disclaimer ────────────────────────────────────────────────
        st.warning(
            "**Disclaimer:** Hasil prediksi bersifat **suportif** dan "
            "**bukan pengganti diagnosis klinis oleh dokter spesialis.** "
            "Gunakan hanya sebagai alat bantu skrining awal."
        )

    else:
        st.info("Silakan unggah citra MRI untuk memulai prediksi.")


# ══════════════════════════════════════════════════════════════════════════
# MODUL 2: DASHBOARD EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════
elif page == "Dashboard Evaluasi Model":
    st.title("Dashboard Evaluasi Model CNN")
    st.markdown(
        "Ringkasan performa model berdasarkan evaluasi pada **data uji (test set)**."
    )

    # ── Tab Layout ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Metrik Evaluasi",
            "Confusion Matrix",
            "Kurva ROC",
            "Kurva Pelatihan",
        ]
    )

    # ── Tab 1: Tabel Metrik ───────────────────────────────────────────
    with tab1:
        st.subheader("Tabel Metrik Evaluasi")

        metrics_df = pd.DataFrame(
            {
                "Metrik": list(METRICS.keys()),
                "Nilai": list(METRICS.values()),
                "Persen": [f"{v * 100:.2f}%" for v in METRICS.values()],
            }
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Ringkasan visual
        if os.path.exists("ringkasan.png"):
            st.subheader("Visualisasi Metrik")
            st.image("ringkasan.png", use_container_width=True)

    # ── Tab 2: Confusion Matrix ───────────────────────────────────────
    with tab2:
        st.subheader("Confusion Matrix - Test Set")

        if os.path.exists("cm.png"):
            st.image("cm.png", use_container_width=True)
        else:
            st.warning("File cm.png tidak ditemukan.")

    # ── Tab 3: Kurva ROC ─────────────────────────────────────────────
    with tab3:
        st.subheader("Kurva ROC - CN vs AD")

        if os.path.exists("roc curve.png"):
            st.image("roc curve.png", use_container_width=True)
        else:
            st.warning("File roc curve.png tidak ditemukan.")

        st.info(
            f"**AUC-ROC = {METRICS['AUC-ROC']:.4f}** - "
            "Kemampuan diskriminasi model sangat baik."
        )

    # ── Tab 4: Kurva Pelatihan ────────────────────────────────────────
    with tab4:
        st.subheader("Kurva Pelatihan Model CNN")

        if os.path.exists("kurva.png"):
            st.image("kurva.png", use_container_width=True)
        else:
            st.warning("File kurva.png tidak ditemukan.")

        st.success(
            "Training berhenti di **epoch 42** (early stopping). "
            "Model terbaik disimpan dari **epoch 38** dengan val_auc = 0.9948."
        )
