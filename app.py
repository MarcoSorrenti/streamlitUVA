import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
import requests
from datetime import datetime

# ============================================================
# METEO API
# ============================================================
def get_weather(lat, lon):
    # Chiamo l'API Open-Meteo
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# ============================================================
# CONFIG
# ============================================================
model_path = "model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
test_json = "dataset/annotations/mimc_test_images.json"

# ============================================================
# STREAMLIT STYLES
# ============================================================
st.set_page_config(page_title="Smart Harvesting", page_icon="🍇", layout="centered")

custom_css = """
<style>

    body {
        background-color: #faf5f0;
    }

    .title-text {
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        margin-bottom: -5px;
        color: #5f2a84;
    }

    .subtitle-text {
        text-align: center;
        font-size: 17px;
        color: #5a3b6e;
        margin-bottom: 25px;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================
# HEADER CON METEO
# ============================================================



# ============================================================
# HEADER CON LOGO e METEO
# ============================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("logo.png", width=65)
with col2:
    st.markdown("<div class='title-text'>Smart Harvesting</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-text'>Carica un'immagine del grappolo per valutarne la maturazione.</div>", unsafe_allow_html=True)

lat, lon = 41.117143, 16.871871

try:
    weather = get_weather(lat, lon)
    daily = weather["daily"]
    dates = daily["time"]
    t_max = daily["temperature_2m_max"]
    t_min = daily["temperature_2m_min"]
    precip = daily["precipitation_probability_max"]

    st.markdown("## Previsioni meteo (oggi + 7 giorni)")

    # Creo le 7 colonne (una per ogni giorno)
    cols = st.columns(7)
    for i in range(7):
        with cols[i]:
            date = datetime.fromisoformat(dates[i]).date()
            # Card con info
            st.markdown(
                f"""
                <div style="
                    background-color: #fff3e0;
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
                ">
                    <strong>{date}</strong><br>
                    Max: {t_max[i]:.1f} °C<br>
                    Min: {t_min[i]:.1f} °C<br>
                    Pioggia: {precip[i]} %
                </div>
                """,
                unsafe_allow_html=True
            )
except Exception as e:
    st.error("Impossibile ottenere le previsioni meteo.")
    st.write(e)

# ============================================================
# COCO CATEGORIES
# ============================================================
coco = COCO(test_json)
cat_ids = sorted(coco.getCatIds())
cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}

# ============================================================
# MODEL
# ============================================================
def load_model(num_classes):
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

num_classes = len(cat_ids)
model = load_model(num_classes)

# ============================================================
# PRED
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(image: Image.Image, threshold=0.5):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        preds = probs > threshold
        labels = [cat_id_to_name[cat_ids[i]] for i, v in enumerate(preds) if v]
    return labels

# ============================================================
# SESSION STATE – Storico
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []  # Lista di tuple (immagine, labels)

# ============================================================
# UI
# ============================================================
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

col_main, col_history = st.columns([3, 1])  # Main content + storico

with col_main:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        if st.button("Valuta", type="primary"):
            labels = predict(image)
            traduzione = {
                "Mature": "Maturo",
                "Semi Mature": "Semi maturo",
                "Immature": "Immaturo"
            }
            labels = [traduzione.get(elem, elem) for elem in labels]

            # Salva nello storico SOLO DOPO LA PREDIZIONE
            st.session_state.history.append((image.copy(), labels))

            # Mostra risultato per l'immagine corrente
            if labels:
                if len(labels) > 1:
                    st.success(f"I grappoli sono: **{', '.join(labels)}**")
                else:
                    st.success(f"Il grappolo è: **{labels[0]}**")
            else:
                st.warning("Non ho rilevato grappoli nell'immagine.")

# ============================================================
# STORICO – MOSTRA LE DUE PRECEDENTI
# ============================================================
with col_history:
    st.markdown("<div class='subtitle-text'>Ultime valutazioni</div>", unsafe_allow_html=True)
    # Mostra le due immagini precedenti, se disponibili
    if len(st.session_state.history) > 1:
        for img_hist, labels_hist in reversed(st.session_state.history[:-1][-2:]):
            st.image(img_hist, use_column_width=True)
            if labels_hist:
                st.caption(", ".join(labels_hist))
            else:
                st.caption("Nessuna valutazione")