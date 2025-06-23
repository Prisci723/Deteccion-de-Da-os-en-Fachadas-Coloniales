import json
import os
import base64
import io
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Codifica una imagen PIL a cadena base64
    
    Args:
        image: Imagen PIL
        
    Returns:
        Cadena base64
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def decode_base64_to_image(base64_str: str) -> Image.Image:
    """
    Decodifica una cadena base64 a imagen PIL
    
    Args:
        base64_str: Cadena base64
        
    Returns:
        Imagen PIL
    """
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def save_results_json(results: Dict[str, Any], filepath: str) -> bool:
    """
    Guarda resultados en formato JSON
    
    Args:
        results: Diccionario con resultados
        filepath: Ruta del archivo
        
    Returns:
        True si se guardó correctamente
    """
    try:
        # Crear una copia de los resultados para poder modificarla
        results_copy = results.copy()
        
        # Convertir objetos no serializables
        for key in results_copy:
            if isinstance(results_copy[key], np.ndarray):
                results_copy[key] = results_copy[key].tolist()
            elif isinstance(results_copy[key], Image.Image):
                results_copy[key] = "<PIL.Image object>"
            elif isinstance(results_copy[key], list):
                # Procesar listas de resultados
                for i, item in enumerate(results_copy[key]):
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v, np.ndarray):
                                results_copy[key][i][k] = v.tolist()
                            elif isinstance(v, Image.Image):
                                results_copy[key][i][k] = "<PIL.Image object>"
        
        # Guardar en archivo
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Resultados guardados en {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar resultados: {str(e)}")
        return False

def load_results_json(filepath: str) -> Dict[str, Any]:
    """
    Carga resultados desde un archivo JSON
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        Diccionario con resultados
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Resultados cargados desde {filepath}")
        return results
    except Exception as e:
        logger.error(f"Error al cargar resultados: {str(e)}")
        return {}