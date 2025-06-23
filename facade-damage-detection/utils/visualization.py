import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any
import io
import base64
import cv2
import logging

logger = logging.getLogger(__name__)

def visualize_results(
    plane_image: Image.Image,
    damage_results: List[Dict[str, Any]],
    segmentation_results: List[Dict[str, Any]],
    architectural_elements: Dict[str, Any]
) -> Image.Image:
    """
    Crea una visualización de los resultados
    
    Args:
        plane_image: Imagen del plano reconstruido
        damage_results: Resultados de detección de daños
        segmentation_results: Resultados de segmentación
        architectural_elements: Resultados de clasificación de elementos
        
    Returns:
        Imagen con visualización de resultados
    """
    # Crear copia de la imagen original
    result_image = plane_image.copy().convert('RGB')
    draw = ImageDraw.Draw(result_image)
    
    # Intentar cargar fuente, si no está disponible usar default
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # 1. Dibujar cajas de daños
    for damage in damage_results:
        if damage["class_name"] != "no_damage" and damage["confidence"] > 0.5:
            coords = damage["coords"]
            x, y = coords["x"], coords["y"]
            w, h = coords["width"], coords["height"]
            
            # Color según tipo de daño (simplificado)
            colors = {
                "damage": (255, 0, 0, 128)  # Rojo semi-transparente
            }
            
            color = colors.get(damage["class_name"], (255, 255, 0, 128))  # Amarillo por defecto
            
            # Dibujar rectángulo semi-transparente
            overlay = Image.new('RGBA', result_image.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.rectangle([x, y, x+w, y+h], fill=color)
            
            # Añadir label con confianza
            label = f"{damage['class_name']}: {damage['confidence']:.2f}"
            draw_overlay.text((x, y-15), label, fill=(255, 255, 255, 200), font=font)
            
            # Combinar con imagen principal
            result_image = Image.alpha_composite(result_image.convert('RGBA'), overlay).convert('RGB')
    
    # 2. Superponer resultados de segmentación si están disponibles
    if segmentation_results:
        for seg_result in segmentation_results:
            # Tomar la imagen de segmentación y aplicarla con cierta transparencia
            seg_image = seg_result.get("segmentation_image")
            if seg_image:
                coords = seg_result.get("coords", {})
                x, y = coords.get("x", 0), coords.get("y", 0)
                
                # Redimensionar si es necesario
                if seg_image.size != (coords.get("width", 224), coords.get("height", 224)):
                    seg_image = seg_image.resize((coords.get("width", 224), coords.get("height", 224)))
                
                # Convertir a RGBA para manejar transparencia
                seg_rgba = seg_image.convert("RGBA")
                
                # Hacer transparente el fondo (clase 0)
                data = np.array(seg_rgba)
                # Donde la imagen es negra (fondo), hacerla transparente
                black_areas = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)
                data[black_areas, 3] = 0
                
                # Para otras áreas, aplicar semi-transparencia
                data[~black_areas, 3] = 128  # 50% de opacidad
                
                seg_transparent = Image.fromarray(data)
                
                # Pegar en la posición correcta
                result_image.paste(seg_transparent, (x, y), seg_transparent)
    
    # 3. Añadir información de elementos arquitectónicos
    if architectural_elements and "elements" in architectural_elements:
        # Añadir texto en la parte superior
        y_pos = 10
        draw = ImageDraw.Draw(result_image)
        draw.rectangle([10, y_pos-5, 300, y_pos + 20*len(architectural_elements["elements"]) + 5], 
                      fill=(0, 0, 0, 180))
        
        draw.text((15, y_pos), "Elementos arquitectónicos detectados:", fill=(255, 255, 255), font=font)
        y_pos += 20
        
        for element in architectural_elements["elements"]:
            text = f"- {element['name']}: {element['confidence']:.2f}"
            draw.text((20, y_pos), text, fill=(255, 255, 255), font=font)
            y_pos += 20
    
    return result_image

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Convierte una imagen PIL a string base64
    
    Args:
        image: Imagen PIL
        
    Returns:
        String base64 codificado
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str