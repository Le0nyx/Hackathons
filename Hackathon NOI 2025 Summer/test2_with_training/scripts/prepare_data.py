import os
import shutil
from sklearn.model_selection import train_test_split

# 1) Definisci il nome delle classi e la corrispondente cartella in data/all_plants
#    ASSICURATI che questi nomi corrispondano ESATTAMENTE alle directory in data/all_plants/
classes = {
    "basil":  "basil",   # es: data/all_plants/basil/
    "tomato": "tomato"   # es: data/all_plants/tomato/
}

# 2) Percorsi principali
src_root = "data/all_plants"
dst_root = "data/basil_tomato"

# 3) Per ciascuna classe, prendi tutte le immagini e crea split train/val
for label, folder in classes.items():
    src_folder = os.path.join(src_root, folder)
    if not os.path.isdir(src_folder):
        raise FileNotFoundError(f"Cartella non trovata: {src_folder}")
    
    # Elenca file (filtra solo immagini .jpg/.png se vuoi)
    all_images = [f for f in os.listdir(src_folder)
                  if os.path.isfile(os.path.join(src_folder, f))
                     and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    # 80% train, 20% validation
    train_imgs, val_imgs = train_test_split(
        all_images, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Copia train
    dst_train = os.path.join(dst_root, "train", label)
    os.makedirs(dst_train, exist_ok=True)
    for img in train_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(dst_train, img)
        )
    
    # Copia validation
    dst_val = os.path.join(dst_root, "val", label)
    os.makedirs(dst_val, exist_ok=True)
    for img in val_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(dst_val, img)
        )

    print(f"[{label}] Copiate {len(train_imgs)} immagini in train/ e {len(val_imgs)} in val/")
