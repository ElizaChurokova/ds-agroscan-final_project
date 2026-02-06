
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# 1. Настройки
MODEL_PATH = "best_plant_model.pth"   # <-- Исправлено имя модели
NUM_CLASSES = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Список классов
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# 2. Трансформации (такие же как при обучении)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    print(f"Загрузка модели из {MODEL_PATH}...")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    else:
        print(f"ОШИБКА: Файл модели {MODEL_PATH} не найден!")
        return None

def predict_custom_image(model, image_path):
    try:
        # Загрузка изображения
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE) # Добавляем batch dimension

        # Предсказание
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence[predicted.item()].item()

        print(f"\n[Файл: {os.path.basename(image_path)}]")
        print(f"   -> Предсказанный класс: {predicted_class}")
        print(f"   -> Уверенность: {confidence_score:.2f}%")

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")

if __name__ == "__main__":
    # Загружаем модель один раз
    model = load_model()
    
    if model:
        # Ищем все картинки в текущей папке
        extensions = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir('.') if f.lower().endswith(extensions)]
        
        print(f"Найдено изображений для теста: {len(files)}")
        
        for img_file in files:
            predict_custom_image(model, img_file)
