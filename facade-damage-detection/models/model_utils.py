
from torchvision import transforms
import torch
import logging
from efficientnet_pytorch import EfficientNet
import timm
import segmentation_models_pytorch as smp
import torch.nn as nn


DAMAGE_TYPES = ['Fondo', 'deformacion', 'desprendimiento', 'deterioro','ensanchamiento','filtracion', 'fisuracion', 'grietas', 'humedad', 'humedad_interna', 'hundimiento']
ARCHITECTURAL_ELEMENTS = ['abertura', 'base_muro', 'espadana', 'muro', 'techo']
DAMAGE_DETECTION_SIZE = 224
DAMAGE_SEGMENTATION_SIZE = 448
ARCHITECTURAL_ELEMENT_SIZE = 228

logger = logging.getLogger(__name__)

def load_test_models():
    """Carga los modelos para la prueba"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    print("El script está corriendo")
    print(device)
    damage_classifier  = timm.create_model('efficientnet_b3', pretrained=False, num_classes=2)
    # num_ftrs = damage_classifier.fc.in_features
    # damage_classifier.fc = nn.Linear(num_ftrs, 2)
    damage_classifier.load_state_dict(torch.load('C:/Users/USER/facade_api/models/weights/best_modelBi_checkpoint150625.pt', map_location=device))
    damage_classifier.to(device)
    damage_classifier.eval()

    damage_segmenter = model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",  # Puedes usar None si no necesitas pesos de imagenet
    in_channels=3,
    classes=11,
    activation=None,
    ).to(device)
    damage_segmenter.load_state_dict(torch.load('C:/Users/USER/facade_api/models/weights/best_model_260525200S.pth',
                                            map_location=device))
    damage_segmenter.to(device)
    damage_segmenter.eval()

    print("\nModelo de segmentación cargado correctamente")


    arch_classifier = EfficientNet.from_name('efficientnet-b0')
    num_ftrs = arch_classifier._fc.in_features
    arch_classifier._fc = nn.Linear(num_ftrs, 5)
    arch_classifier.load_state_dict(torch.load('C:/Users/USER/facade_api/models/weights/best_model_checkpoint_150625.pt',
                                          map_location=device))
    arch_classifier.to(device)
    arch_classifier.eval()
    print("Modelo de elementos arquitectónicos cargado correctamente")

    logger.info("Modelos cargados correctamente")
    
    return damage_classifier, damage_segmenter, arch_classifier

def get_transforms():
    # Transformación para el modelo de detección de daños
    damage_detection_transform = transforms.Compose([
        transforms.Resize((DAMAGE_DETECTION_SIZE, DAMAGE_DETECTION_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformación para el modelo de segmentación
    damage_segmentation_transform = transforms.Compose([
        transforms.Resize((DAMAGE_SEGMENTATION_SIZE, DAMAGE_SEGMENTATION_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformación para el modelo de elementos arquitectónicos
    architectural_elements_transform = transforms.Compose([
        transforms.Resize((ARCHITECTURAL_ELEMENT_SIZE, ARCHITECTURAL_ELEMENT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return damage_detection_transform, damage_segmentation_transform, architectural_elements_transform
