import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Any
import logging
import base64
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import json

from models.all_models_2 import ImprovedConfig

logger = logging.getLogger(__name__)

def generate_patches(image, patch_size, stride):
    """
    Genera patches de una imagen usando sliding window
    
    Args:
        image: imagen numpy (H, W, C)
        patch_size: tamaño del patch (int)
        stride: paso entre patches (int)
        
    Returns:
        patches: lista de arrays numpy
        locations: lista de tuplas (x, y)
    """
    patches = []
    locations = []
    height, width = image.shape[:2]

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            locations.append((x, y))

    return patches, locations

def split_image_into_patches(image: Image.Image, patch_size: Tuple[int, int] = (224, 224), overlap: int = 56) -> List[Tuple[Image.Image, Dict[str, Any]]]:
    """
    Divide una imagen en patches con overlap
    
    Args:
        image: Imagen PIL
        patch_size: tamaño del patch (ancho, alto)
        overlap: cantidad de solapamiento entre patches
        
    Returns:
        Lista de tuplas (patch_image, metadata) donde metadata tiene las coordenadas
    """
    if not isinstance(image, Image.Image):
        raise TypeError("La imagen debe ser un objeto PIL.Image")
    
    # Convertir a numpy para procesamiento
    np_image = np.array(image)
    
    # Calcular stride (paso)
    stride = patch_size[0] - overlap
    
    # Generar patches y ubicaciones
    patch_arrays, locations = generate_patches(np_image, patch_size[0], stride)
    
    # Convertir de nuevo a PIL y añadir metadatos
    result = []
    for i, (patch_array, loc) in enumerate(zip(patch_arrays, locations)):
        patch_pil = Image.fromarray(patch_array)
        metadata = {
            "index": i,
            "x": loc[0],
            "y": loc[1],
            "width": patch_size[0],
            "height": patch_size[1]
        }
        result.append((patch_pil, {"coords": metadata}))
    
    logger.info(f"Imagen dividida en {len(result)} patches de {patch_size}")
    return result

def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Redimensiona una imagen manteniendo la relación de aspecto
    
    Args:
        image: Imagen PIL
        max_size: tamaño máximo (ancho o alto)
        
    Returns:
        Imagen redimensionada
    """
    width, height = image.size
    
    # Calcular nueva dimensión manteniendo relación de aspecto
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def base64_to_cv2(base64_string):
    """Convierte una imagen base64 a formato OpenCV (BGR)"""
    try:
        # Remover el prefijo data:image/...;base64, si existe
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
       
        # Decodificar base64
        img_data = base64.b64decode(base64_string)
       
        # Convertir a numpy array
        nparr = np.frombuffer(img_data, np.uint8)
       
        # Decodificar la imagen (OpenCV la lee en BGR por defecto)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        if img is None:
            raise ValueError("No se pudo decodificar la imagen")
           
        return img
    except Exception as e:
        raise ValueError(f"Error al convertir base64 a imagen: {str(e)}")

def base64_to_rgb(base64_string):
    """Convierte una imagen base64 a formato RGB (para usar con PIL/torchvision)"""
    try:
        # Primero convertir a OpenCV (BGR)
        img_bgr = base64_to_cv2(base64_string)
        
        # Convertir de BGR a RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    except Exception as e:
        raise ValueError(f"Error al convertir base64 a RGB: {str(e)}")

def cv2_to_base64(image):
    """Convierte una imagen OpenCV (BGR) a base64"""
    try:
        # Codificar imagen a formato JPEG
        _, buffer = cv2.imencode('.jpg', image)
       
        # Convertir a base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
       
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        raise ValueError(f"Error al convertir imagen a base64: {str(e)}")

def rgb_to_base64(image_rgb):
    """Convierte una imagen RGB (numpy array) a base64"""
    try:
        # Convertir de RGB a BGR para OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return cv2_to_base64(image_bgr)
    except Exception as e:
        raise ValueError(f"Error al convertir RGB a base64: {str(e)}")

def get_optimal_image_params(original_size_mb, tunnel_limit_mb=8):
    """
    Calcula parámetros óptimos de compresión basados en el tamaño original
    """
    # Dejar margen de seguridad (usar 80% del límite)
    safe_limit = tunnel_limit_mb * 0.8
    
    if original_size_mb <= safe_limit:
        return {
            'max_size_mb': safe_limit,
            'max_dimension': 3072,
            'quality_start': 95
        }
    elif original_size_mb <= safe_limit * 2:
        return {
            'max_size_mb': safe_limit * 0.8,
            'max_dimension': 2048,
            'quality_start': 85
        }
    else:
        return {
            'max_size_mb': safe_limit * 0.6,
            'max_dimension': 1536,
            'quality_start': 75
        }

def progressive_compression(image, target_size_mb=2):
    """
    Compresión progresiva más sofisticada
    """
    import io
    from PIL import Image
    
    pil_image = Image.fromarray(image.astype('uint8'))
    
    # Estrategia de compresión en múltiples pasos
    strategies = [
        # (max_dimension, quality_range)
        (3072, range(95, 70, -5)),    # Alta calidad, dimensión completa
        (2560, range(90, 60, -5)),    # Buena calidad, ligera reducción
        (2048, range(85, 50, -5)),    # Calidad media, reducción moderada
        (1536, range(80, 40, -5)),    # Calidad baja, reducción significativa
        (1024, range(70, 30, -5)),    # Calidad muy baja, máxima reducción
    ]
    
    original_size = pil_image.size
    
    for max_dim, quality_range in strategies:
        # Redimensionar si es necesario
        if max(original_size) > max_dim:
            ratio = max_dim / max(original_size)
            new_size = tuple(int(dim * ratio) for dim in original_size)
            working_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            working_image = pil_image
        
        # Probar diferentes calidades
        for quality in quality_range:
            buffer = io.BytesIO()
            working_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            size_mb = len(buffer.getvalue()) / (1024 * 1024)
            
            if size_mb <= target_size_mb:
                buffer.seek(0)
                return buffer.getvalue(), size_mb, quality, working_image.size
    
    # Si no se pudo comprimir lo suficiente, devolver la última versión
    return buffer.getvalue(), size_mb, quality, working_image.size

def get_adaptive_config_for_panorama(image_shape: tuple) -> ImprovedConfig:
    """
    Configuración adaptativa específica para tu sistema
    """
    height, width = image_shape[:2]
    total_pixels = height * width
    
    if total_pixels > 4000000:  # Imagen muy grande (>4MP)
        return ImprovedConfig(
            damage_detection_threshold=0.5,  # Más conservador
            damage_segmentation_threshold=0.7,
            architectural_elements_threshold=0.65,
            batch_size=16,  # Batch menor para memoria
            overlap_ratio=0.2,  # Menos solapamiento
            save_intermediate_results=False,
            visualization_dpi=150
        )
    elif total_pixels > 1000000:  # Imagen grande (>1MP)
        return ImprovedConfig(
            damage_detection_threshold=0.5,  # Balanceado
            damage_segmentation_threshold=0.65,
            architectural_elements_threshold=0.6,
            batch_size=32,
            overlap_ratio=0.25,  # Tu valor original
            save_intermediate_results=False,
            visualization_dpi=200
        )
    else:  # Imagen pequeña
        return ImprovedConfig(
            damage_detection_threshold=0.5,  # Más sensible
            damage_segmentation_threshold=0.6,
            architectural_elements_threshold=0.55,
            batch_size=64,  # Batch mayor
            overlap_ratio=0.3,  # Más solapamiento para mejor precisión
            save_intermediate_results=True,
            visualization_dpi=300
        )
