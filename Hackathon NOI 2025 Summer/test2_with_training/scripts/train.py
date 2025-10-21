# scripts/train.py
"""
Script di training per il classificatore basilico vs pomodoro.
Funzionalità:
  - Carica dataset da data/basil_tomato/train e /val
  - Transfer learning con EfficientNet-B0
  - Salva il miglior modello in models/basil_tomato_classifier.pth
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from pathlib import Path

# 1) Percorsi dataset (usa percorsi assoluti per sicurezza)
script_dir = Path(__file__).parent
base_dir = script_dir.parent if script_dir.parent.name != "scripts" else script_dir
train_dir = base_dir / "scripts" / "data" / "basil_tomato" / "train"
val_dir = base_dir / "scripts" / "data" / "basil_tomato" / "val"
models_dir = base_dir / "scripts" / "models"

print(f"🔍 Cercando dataset in:")
print(f"   Train: {train_dir}")
print(f"   Val: {val_dir}")
print(f"   Models: {models_dir}")

# Verifica esistenza directory
if not train_dir.exists():
    print(f"❌ Directory train non trovata: {train_dir}")
    sys.exit(1)
if not val_dir.exists():
    print(f"❌ Directory validation non trovata: {val_dir}")
    sys.exit(1)

# 2) Valori standard di normalizzazione ImageNet
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD  = [0.229, 0.224, 0.225]

# 3) Trasformazioni dati
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
])

# 4) Dataset e DataLoader con error handling
try:
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_transforms)
    
    if len(train_ds) == 0:
        print(f"❌ Nessuna immagine trovata in {train_dir}")
        sys.exit(1)
    if len(val_ds) == 0:
        print(f"❌ Nessuna immagine trovata in {val_dir}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Errore nel caricamento dataset: {e}")
    sys.exit(1)

# Ottimizza batch size per GPU disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    batch_size = 32 if gpu_memory > 6 else 16
    num_workers = min(4, os.cpu_count() or 1)
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)")
    print(f"⚙️  Batch size ottimizzato: {batch_size}")
else:
    batch_size = 8
    num_workers = min(2, os.cpu_count() or 1)
    print("💻 Usando CPU")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=torch.cuda.is_available()
)

print(f"✅ Classi trovate: {train_ds.classes}")
print(f"📊 Numero immagini - Train: {len(train_ds)}, Validation: {len(val_ds)}")

# Verifica bilanciamento classi
class_counts_train = {}
class_counts_val = {}
for idx, (_, label) in enumerate(train_ds):
    class_name = train_ds.classes[label]
    class_counts_train[class_name] = class_counts_train.get(class_name, 0) + 1
for idx, (_, label) in enumerate(val_ds):
    class_name = val_ds.classes[label]
    class_counts_val[class_name] = class_counts_val.get(class_name, 0) + 1

print(f"📈 Distribuzione train: {class_counts_train}")
print(f"📈 Distribuzione val: {class_counts_val}")

# 5) Configura device (già fatto sopra)

# 6) Costruisci il modello con pesi pre-addestrati (fix deprecation warning)
print("🔄 Caricando EfficientNet-B0 con pesi pre-addestrati...")
try:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_classes = len(train_ds.classes)
    
    # Sostituisci il classificatore finale
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    model = model.to(device)
    
    # Ottimizzazioni per GPU
    if torch.cuda.is_available():
        model = model.half()  # Usa mixed precision per risparmiare memoria
        print("✅ Mixed precision attivata")
    
    print(f"✅ Modello caricato con {num_classes} classi: {train_ds.classes}")
    
except Exception as e:
    print(f"❌ Errore nel caricamento del modello: {e}")
    sys.exit(1)

# 7) Criterio e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=1e-4, weight_decay=1e-5
)

# 8) Funzione di training per un'epoca con progress tracking
def train_epoch():
    model.train()
    running_loss, running_corrects = 0.0, 0
    total_batches = len(train_loader)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Mixed precision per GPU
        if torch.cuda.is_available():
            inputs = inputs.half()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (outputs.argmax(1) == labels).sum().item()
        
        # Progress tracking
        if (batch_idx + 1) % max(1, total_batches // 10) == 0:
            progress = (batch_idx + 1) / total_batches * 100
            print(f"   📈 Training progress: {progress:.1f}% ({batch_idx + 1}/{total_batches})")

    epoch_loss = running_loss / len(train_ds)
    epoch_acc = running_corrects / len(train_ds)
    return epoch_loss, epoch_acc

# 9) Funzione di validazione con progress tracking
def validate_epoch():
    model.eval()
    val_loss, val_corrects = 0.0, 0
    total_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixed precision per GPU
            if torch.cuda.is_available():
                inputs = inputs.half()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += (outputs.argmax(1) == labels).sum().item()
            
            # Progress tracking
            if (batch_idx + 1) % max(1, total_batches // 5) == 0:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"   📊 Validation progress: {progress:.1f}% ({batch_idx + 1}/{total_batches})")

    loss = val_loss / len(val_ds)
    acc = val_corrects / len(val_ds)
    return loss, acc

# 10) Loop di training principale con miglioramenti
if __name__ == "__main__":
    import time
    
    best_val_acc = 0.0
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n🚀 Iniziando training per {10} epoche...")
    print(f"💾 I modelli saranno salvati in: {models_dir}")
    
    start_time = time.time()

    for epoch in range(1, 11):  # 10 epoche
        epoch_start = time.time()
        print(f"\n🔄 Epoca {epoch}/10:")
        
        # Training
        print("   🏋️  Training...")
        train_loss, train_acc = train_epoch()
        
        # Validation
        print("   🔍 Validation...")
        val_loss, val_acc = validate_epoch()
        
        epoch_time = time.time() - epoch_start

        print(
            f"✅ Epoca {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f} | tempo={epoch_time:.1f}s"
        )

        # Salva il miglior modello con validazione
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = models_dir / "basil_tomato_classifier.pth"
            
            try:
                # Salva sia state_dict che modello completo
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'classes': train_ds.classes,
                    'num_classes': num_classes
                }, save_path)
                
                print(f"💾 Nuovo best model salvato con val_acc={val_acc:.4f}")
                
            except Exception as e:
                print(f"❌ Errore nel salvataggio: {e}")
        
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\n🎉 Training completato! Best val_acc: {best_val_acc:.4f}")
    print(f"⏱️  Tempo totale: {total_time:.1f}s ({total_time/60:.1f} minuti)")
    
    # Statistiche finali
    if torch.cuda.is_available():
        print(f"📊 Memoria GPU utilizzata: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
