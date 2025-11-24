import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO

# ============================================================
# CONFIGURAZIONE
# ============================================================
model_path = "model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
test_json = "dataset/annotations/mimc_test_images.json"

# ============================================================
# CARICAMENTO CATEGORIE COCO
# ============================================================
coco = COCO(test_json)
cat_ids = sorted(coco.getCatIds())
cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}

# ============================================================
# CARICAMENTO MODELLO (ResNet50)
# ============================================================
def load_model(num_classes):
    from torchvision.models import resnet50
    model = resnet50(weights=None)  # niente pesi pretrained, li carichiamo
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Carica pesi addestrati
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model

num_classes = len(cat_ids)
model = load_model(num_classes)

# ============================================================
# PREPROCESSING E INFERENZA
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
# STREAMLIT GUI
# ============================================================
st.title("Smart Harvesting")
st.write("Carica un'immagine o una foto, ed io ti dirò se il grappolo è Immaturo, Semi Maturo, Maturo.")

uploaded_file = st.file_uploader("Scegli un'immagine", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Immagine selezionata", use_container_width=True)
    
    if st.button("Valuta"):
        labels = predict(image)
        print(type(labels))
        if labels:
            if len(labels)>1:
                st.success(f"I grappoli sono: {', '.join(labels)}")
            else:
                st.success(f"Il grappolo è: {', '.join(labels)}")

        else:
            st.warning("Oooops, non ho rilevato grappoli nell'immagine.")
