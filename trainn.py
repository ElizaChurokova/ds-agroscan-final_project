import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk
from torch.cuda.amp import GradScaler, autocast # Для ускорения на RTX 4060

# --- 1. Улучшенные гиперпараметры ---
BATCH_SIZE = 32
EPOCHS = 15 # Увеличим немного количество эпох
LEARNING_RATE = 0.0005 # Начнем с чуть меньшего шага для стабильности
NUM_CLASSES = 38 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Улучшенная аугментация ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Если фото разного освещения
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Для теста/валидации аугментация НЕ нужна, только ресайз
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def transform_train(examples):
    examples["pixel_values"] = [train_transform(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def transform_val(examples):
    examples["pixel_values"] = [val_transform(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

# Загрузка и разбиение
dataset = load_from_disk('PlantVillage_Dataset_2026-01-21')
if "test" not in dataset: # Если нет разделения, сделаем сами
    dataset = dataset.train_test_split(test_size=0.2)

train_data = dataset["train"].with_transform(transform_train)
val_data = dataset["test"].with_transform(transform_val)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# --- 3. Модель и Оптимизация ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(DEVICE)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
# Снижаем LR в 10 раз, если лосс не падает 2 эпохи
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler() # Для Mixed Precision

# --- 4. Улучшенный Цикл Обучения ---
best_acc = 0.0



for epoch in range(EPOCHS):
    # ФАЗА ОБУЧЕНИЯ
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
    
    for batch in loop:
        images, labels = batch["pixel_values"].to(DEVICE), batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        with autocast(): # Запуск в режиме ускорения (Mixed Precision)
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # ФАЗА ВАЛИДАЦИИ (Проверка качества)
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[VAL]"):
            images, labels = batch["pixel_values"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    print(f"--> Val Acc: {acc:.2f}% | Avg Loss: {avg_val_loss:.4f}")
    
    # Сохраняем лучшую версию
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_plant_model.pth")
        print("!!! New Best Model Saved !!!")
    
    scheduler.step(avg_val_loss) # Обновляем скорость обучения