# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# from pycocotools.coco import COCO

# # ============================================================
# # CONFIGURAZIONE
# # ============================================================
# model_path = "model.pt"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# test_json = "dataset/annotations/mimc_test_images.json"

# # ============================================================
# # CARICAMENTO CATEGORIE COCO
# # ============================================================
# coco = COCO(test_json)
# cat_ids = sorted(coco.getCatIds())
# cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}

# # ============================================================
# # CARICAMENTO MODELLO (ResNet50)
# # ============================================================
# def load_model(num_classes):
#     from torchvision.models import resnet50
#     model = resnet50(weights=None)  # niente pesi pretrained, li carichiamo
#     model.fc = nn.Linear(model.fc.in_features, num_classes)

#     # Carica pesi addestrati
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict, strict=True)

#     model.to(device)
#     model.eval()
#     return model

# num_classes = len(cat_ids)
# model = load_model(num_classes)

# # ============================================================
# # PREPROCESSING E INFERENZA
# # ============================================================
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# def predict(image: Image.Image, threshold=0.5):
#     img = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(img)
#         probs = torch.sigmoid(outputs).cpu().numpy()[0]
#         preds = probs > threshold
#         labels = [cat_id_to_name[cat_ids[i]] for i, v in enumerate(preds) if v]
#     return labels

# # ============================================================
# # STREAMLIT GUI
# # ============================================================
# st.title("Smart Harvesting")
# st.write("Carica un'immagine o una foto, ed io ti dirò se il grappolo è Immaturo, Semi Maturo, Maturo.")

# uploaded_file = st.file_uploader("Scegli un'immagine", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Immagine selezionata")
    
#     if st.button("Valuta"):
#         labels = predict(image)
#         traduzione = {
#             "Mature": "Maturo",
#             "Semi Mature": "Semi maturo",
#             "Immature": "Immaturo"
#         }
#         labels = [traduzione.get(elem, elem) for elem in labels]
        
#         if labels:
#             if len(labels)>1:
#                 st.success(f"I grappoli sono: {', '.join(labels)}")
#             else:
#                 st.success(f"Il grappolo è: {', '.join(labels)}")

#         else:
#             st.warning("Oooops, non ho rilevato grappoli nell'immagine.")


import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO

# ============================================================
# CONFIG
# ============================================================
model_path = "model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
test_json = "dataset/annotations/mimc_test_images.json"


# ============================================================
# STREAMLIT STILE
# ============================================================
st.set_page_config(page_title="Smart Harvesting", page_icon="🍇", layout="centered")

custom_css = """
<style>

    body { background-color: #faf5f0; }

    .title-text {
        font-size: 42px;
        font-weight: 800;
        color: #6a0dad;
        text-align: center;
        margin-bottom: -10px;
    }

    .subtitle-text {
        text-align: center;
        font-size: 18px;
        color: #4a235a;
        margin-bottom: 25px;
    }

    /* Card per l'uploader */
    .stFileUploader {
        padding: 25px !important;
        background: #ffffff !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.09) !important;
        border: 1px solid #e0d4f7 !important;
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Card del risultato */
    .result-box {
        padding: 20px;
        background: #f6ecff;
        border-radius: 15px;
        border: 1px solid #d3c0f5;
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Immagine più piccola */
    .stImage img {
        width: 250px !important;
        border-radius: 12px !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ============================================================
# LOGO + TITOLO
# ============================================================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=70)

st.markdown("<div class='title-text'>Smart Harvesting</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Carica un'immagine del grappolo e rileverò il livello di maturazione.</div>", unsafe_allow_html=True)


# ============================================================
# COCO
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
# PREDIZIONE
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
# UPLOADER (IN UNA CARD STILIZZATA)
# ============================================================

uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Anteprima immagine")

    if st.button("🔍 Valuta", type="primary"):
        labels = predict(image)

        traduzione = {
            "Mature": "Maturo",
            "Semi Mature": "Semi maturo",
            "Immature": "Immaturo"
        }
        labels = [traduzione.get(elem, elem) for elem in labels]

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        if labels:
            if len(labels) > 1:
                st.success(f"I grappoli sono: {', '.join(labels)}")
            else:
                st.success(f"Il grappolo è: {labels[0]}")
        else:
            st.warning("Oooops, non ho rilevato grappoli nell'immagine.")

        st.markdown("</div>", unsafe_allow_html=True)

