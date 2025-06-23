import base64
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import PIL.Image
from PIL import Image

DAMAGE_TYPES = ['Fondo', 'deformacion', 'desprendimiento', 'deterioro','ensanchamiento','filtracion', 'fisuracion', 'grietas', 'humedad', 'humedad_interna', 'hundimiento']
ARCHITECTURAL_ELEMENTS = ['abertura', 'base_muro', 'espadana', 'muro', 'techo']
DAMAGE_DETECTION_SIZE = 224
DAMAGE_SEGMENTATION_SIZE = 448
ARCHITECTURAL_ELEMENT_SIZE = 228
DAMAGE_COLORS = np.array([
    [0, 0, 0],       # Fondo - Negro
    [255, 0, 0],     # deformacion - Rojo
    [0, 0, 255],     # desprendimiento - Azul
    [255, 255, 0],   # deterioro - Amarillo
    [255, 0, 255],   # ensanchamiento - Magenta
    [0, 255, 0],     # filtracion - Verde
    [0, 255, 155],   # fisuracion - Verde azulado
    [128, 0, 0],     # grietas - Marrón oscuro
    [0, 128, 128],   # humedad - Turquesa oscuro
    [128, 128, 0],   # humedad_interna - Oliva
    [128, 0, 128],   # hundimiento - Púrpura oscuro
])

def visualize_results_improved(facade_img, facade_visualization, damage_mask, element_mask):
    """
    Versión mejorada de tu función de visualización original
    """
    # Crear la figura
    fig = plt.figure(figsize=(20, 15))

    # 1. Imagen original
    plt.subplot(2, 2, 1)
    plt.title("Imagen Original de la Fachada", fontsize=14, fontweight='bold')
    plt.imshow(facade_img)
    plt.axis('off')

    # 2. Detección de daños (patches marcados con confianza)
    plt.subplot(2, 2, 2)
    plt.title("Detección de Daños Mejorada (con Confianza)", fontsize=14, fontweight='bold')
    plt.imshow(facade_visualization)
    plt.axis('off')

    # 3. Mapa de segmentación de daños
    plt.subplot(2, 2, 3)
    plt.title("Segmentación de Tipos de Daños", fontsize=14, fontweight='bold')

    # Crear una imagen coloreada para la segmentación
    damage_vis = np.zeros((*damage_mask.shape, 3), dtype=np.uint8)
    for i in range(1, len(DAMAGE_TYPES)):
        if i < len(DAMAGE_COLORS):
            damage_vis[damage_mask == i] = DAMAGE_COLORS[i]

    # Superponer la segmentación sobre la imagen original
    alpha = 0.7
    if facade_img.dtype != np.uint8:
        facade_img_uint8 = (facade_img * 255).astype(np.uint8) if facade_img.max() <= 1.0 else facade_img.astype(np.uint8)
    else:
        facade_img_uint8 = facade_img
    
    damage_overlay = cv2.addWeighted(facade_img_uint8, 1-alpha, damage_vis, alpha, 0)
    plt.imshow(damage_overlay)

    # Añadir leyenda mejorada para tipos de daños
    damage_patches_legend = []
    for i in range(1, len(DAMAGE_TYPES)):
        if i < len(DAMAGE_COLORS) and np.any(damage_mask == i):
            damage_patches_legend.append(
                patches.Patch(color=np.array(DAMAGE_COLORS[i])/255.0, label=DAMAGE_TYPES[i])
            )
    
    if damage_patches_legend:
        plt.legend(handles=damage_patches_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')

    # 4. Mapa de elementos arquitectónicos
    plt.subplot(2, 2, 4)
    plt.title("Elementos Arquitectónicos en Áreas Dañadas", fontsize=14, fontweight='bold')

    # Crear un mapa de colores para los elementos
    cmap = plt.cm.get_cmap('Set3', len(ARCHITECTURAL_ELEMENTS))
    element_vis = np.zeros((*element_mask.shape, 3), dtype=np.uint8)
    
    for i in range(1, len(ARCHITECTURAL_ELEMENTS)):
        if np.any(element_mask == i):
            color = np.array(cmap(i)[:3]) * 255
            element_vis[element_mask == i] = color.astype(np.uint8)

    # Superponer los elementos sobre la imagen original
    alpha = 0.6
    facade_normalized = facade_img_uint8 / 255.0 if facade_img_uint8.max() > 1.0 else facade_img_uint8
    element_normalized = element_vis / 255.0
    
    element_overlay = cv2.addWeighted(facade_normalized.astype(np.float32), 1-alpha, 
                                    element_normalized.astype(np.float32), alpha, 0)
    plt.imshow(element_overlay)

    # Añadir leyenda para elementos arquitectónicos
    element_patches_legend = []
    for i in range(1, len(ARCHITECTURAL_ELEMENTS)):
        if np.any(element_mask == i):
            element_patches_legend.append(
                patches.Patch(color=cmap(i)[:3], label=ARCHITECTURAL_ELEMENTS[i])
            )
    
    if element_patches_legend:
        plt.legend(handles=element_patches_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')

    plt.tight_layout()
    
    # Guardar en memoria como imagen
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    
    # Convertir a PIL Image
    result_image = PIL.Image.open(buffer)
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)
    
    return result_image

# Función mejorada para crear imágenes individuales con leyendas
def create_individual_analysis_images(facade_img, facade_visualization, damage_mask, element_mask):
    """
    Crea imágenes individuales para cada tipo de análisis
    Retorna un diccionario con las imágenes por separado (todas como arrays numpy)
    """
    images = {}
    
    # Asegurar que facade_img esté en el formato correcto
    if facade_img.dtype != np.uint8:
        facade_img_uint8 = (facade_img * 255).astype(np.uint8) if facade_img.max() <= 1.0 else facade_img.astype(np.uint8)
    else:
        facade_img_uint8 = facade_img
    
    # 1. Imagen de detección de daños (patches marcados con confianza)
    images['damage_detection'] = facade_visualization
    
    # 2. Imagen de segmentación de daños CON LEYENDA
    # Crear una imagen coloreada para la segmentación
    damage_vis = np.zeros((*damage_mask.shape, 3), dtype=np.uint8)
    for i in range(1, len(DAMAGE_TYPES)):
        if i < len(DAMAGE_COLORS):
            damage_vis[damage_mask == i] = DAMAGE_COLORS[i]
    
    # Superponer la segmentación sobre la imagen original
    alpha = 0.7
    damage_overlay = cv2.addWeighted(facade_img_uint8, 1-alpha, damage_vis, alpha, 0)
    
    # Crear figura para la imagen con leyenda
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(damage_overlay)
    ax.set_title("Segmentación de Tipos de Daños", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Crear leyenda solo para los tipos de daños que están presentes en la imagen
    damage_patches_legend = []
    for i in range(1, len(DAMAGE_TYPES)):
        if i < len(DAMAGE_COLORS) and np.any(damage_mask == i):
            damage_patches_legend.append(
                patches.Patch(color=np.array(DAMAGE_COLORS[i])/255.0, label=DAMAGE_TYPES[i])
            )
    
    if damage_patches_legend:
        ax.legend(handles=damage_patches_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Guardar la imagen con leyenda en memoria
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    
    # Convertir a PIL Image y luego a array numpy
    segmentation_pil = PIL.Image.open(buffer)
    segmentation_array = np.array(segmentation_pil)
    
    # Convertir RGBA a RGB si es necesario
    if segmentation_array.shape[-1] == 4:
        segmentation_array = segmentation_array[:, :, :3]
    
    images['damage_segmentation'] = segmentation_array
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)
    
    return images

# Función adicional para crear solo la imagen de segmentación con leyenda
def create_damage_segmentation_with_legend(facade_img, damage_mask):
    """
    Crea únicamente la imagen de segmentación de daños con su leyenda
    """
    # Asegurar que facade_img esté en el formato correcto
    if facade_img.dtype != np.uint8:
        facade_img_uint8 = (facade_img * 255).astype(np.uint8) if facade_img.max() <= 1.0 else facade_img.astype(np.uint8)
    else:
        facade_img_uint8 = facade_img
    
    # Crear una imagen coloreada para la segmentación
    damage_vis = np.zeros((*damage_mask.shape, 3), dtype=np.uint8)
    for i in range(1, len(DAMAGE_TYPES)):
        if i < len(DAMAGE_COLORS):
            damage_vis[damage_mask == i] = DAMAGE_COLORS[i]
    
    # Superponer la segmentación sobre la imagen original
    alpha = 0.7
    damage_overlay = cv2.addWeighted(facade_img_uint8, 1-alpha, damage_vis, alpha, 0)
    
    # Crear figura para la imagen con leyenda
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(damage_overlay)
    ax.set_title("Segmentación de Tipos de Daños", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Crear leyenda solo para los tipos de daños que están presentes en la imagen
    damage_patches_legend = []
    for i in range(1, len(DAMAGE_TYPES)):
        if i < len(DAMAGE_COLORS) and np.any(damage_mask == i):
            damage_patches_legend.append(
                patches.Patch(color=np.array(DAMAGE_COLORS[i])/255.0, label=DAMAGE_TYPES[i])
            )
    
    if damage_patches_legend:
        ax.legend(handles=damage_patches_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Guardar la imagen con leyenda en memoria
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    
    # Convertir a PIL Image
    result_image = PIL.Image.open(buffer)
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)
    
    return result_image

def compress_image_for_api(image, max_size_mb=2, max_dimension=2048):
    """
    Comprime una imagen para que sea adecuada para transmisión API
    """
    # Convertir de OpenCV (BGR) a PIL (RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Asumiendo que viene en RGB desde tu visualización
        pil_image = Image.fromarray(image.astype('uint8'))
    else:
        pil_image = Image.fromarray(image)
    
    # Redimensionar si es necesario
    width, height = pil_image.size
    if width > max_dimension or height > max_dimension:
        # Mantener proporción
        ratio = min(max_dimension/width, max_dimension/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"📏 Imagen redimensionada de {width}x{height} a {new_width}x{new_height}")
    
    # Comprimir con calidad variable hasta alcanzar el tamaño objetivo
    for quality in [95, 85, 75, 65, 55, 45, 35]:
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        
        print(f"🎯 Calidad {quality}%: {size_mb:.2f}MB")
        
        if size_mb <= max_size_mb:
            buffer.seek(0)
            compressed_bytes = buffer.getvalue()
            print(f"✅ Imagen comprimida exitosamente: {size_mb:.2f}MB (calidad {quality}%)")
            return compressed_bytes
    
    # Si no pudimos comprimir lo suficiente, usar la última calidad
    print(f"⚠️ Imagen final: {size_mb:.2f}MB (calidad mínima {quality}%)")
    return compressed_bytes

def compress_and_encode_images(images_dict, original_size_info):
    """
    Comprime y codifica múltiples imágenes a base64
    """
    encoded_images = {}
    compression_info = {}
    
    for image_name, image in images_dict.items():
        try:
            # Determinar límite basado en el tamaño original
            original_size_mb = (image.nbytes) / (1024 * 1024)
            print(f"📊 Tamaño original de {image_name}: {original_size_mb:.2f}MB")
            
            # Límite adaptativo: más estricto para imágenes muy grandes
            if original_size_mb > 10:
                max_size_limit = 1.5  # 1.5MB para imágenes muy grandes
                max_dimension_limit = 1536
            elif original_size_mb > 5:
                max_size_limit = 2.0  # 2MB para imágenes grandes
                max_dimension_limit = 2048
            else:
                max_size_limit = 3.0  # 3MB para imágenes normales
                max_dimension_limit = 2560
            
            compressed_image_bytes = compress_image_for_api(
                image, 
                max_size_mb=max_size_limit,
                max_dimension=max_dimension_limit
            )
            
            # Convertir a base64
            encoded_images[image_name] = base64.b64encode(compressed_image_bytes).decode('utf-8')
            
            # Calcular información de compresión
            final_size_mb = len(encoded_images[image_name].encode('utf-8')) / (1024 * 1024)
            compression_ratio = (original_size_mb / final_size_mb) if final_size_mb > 0 else 0
            
            compression_info[image_name] = {
                'original_size_mb': round(original_size_mb, 2),
                'final_size_mb': round(final_size_mb, 2),
                'compression_ratio': round(compression_ratio, 1)
            }
            
            print(f"📦 {image_name} final: {final_size_mb:.2f}MB (reducción: {compression_ratio:.1f}x)")
            
        except Exception as e:
            print(f"⚠️ Error en compresión de {image_name}, usando método alternativo: {str(e)}")
            # Fallback: usar método más simple
            try:
                pil_image = Image.fromarray(image.astype('uint8'))
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=75)
                encoded_images[image_name] = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                final_size_mb = len(encoded_images[image_name].encode('utf-8')) / (1024 * 1024)
                compression_info[image_name] = {
                    'original_size_mb': round(original_size_mb, 2),
                    'final_size_mb': round(final_size_mb, 2),
                    'compression_ratio': 1.0
                }
            except Exception as e2:
                print(f"❌ Error crítico en {image_name}: {str(e2)}")
                encoded_images[image_name] = None
                compression_info[image_name] = {'error': str(e2)}
    
    return encoded_images, compression_info
