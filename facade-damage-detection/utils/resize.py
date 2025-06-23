import cv2
import numpy as np
import math
from typing import Tuple, Optional

def smart_resize_for_patches(image, target_base_size=(1280, 720), patch_size=224,
                           method='maintain_scale', max_size=None, min_size=None):
    """
    Redimensiona una imagen de manera inteligente para análisis de patches.

    Args:
        image: Imagen de entrada (numpy array)
        target_base_size: Tamaño base usado en entrenamiento (width, height)
        patch_size: Tamaño del patch (224)
        method: Método de redimensionamiento
            - 'maintain_scale': Mantiene la escala de elementos del entrenamiento
            - 'fit_to_multiple': Ajusta a múltiplos exactos del patch_size
            - 'preserve_aspect': Preserva aspect ratio manteniendo tamaño manejable
            - 'adaptive': Combina estrategias según el tamaño de entrada
        max_size: Tamaño máximo permitido (width, height)
        min_size: Tamaño mínimo permitido (width, height)

    Returns:
        resized_image: Imagen redimensionada
        scale_factor: Factor de escala aplicado (útil para análisis)
        final_size: Tamaño final (width, height)
    """

    if image is None:
        raise ValueError("La imagen no puede ser None")

    original_height, original_width = image.shape[:2]
    original_size = (original_width, original_height)
    target_width, target_height = target_base_size

    print(f"Imagen original: {original_width}x{original_height}")
    print(f"Tamaño base de entrenamiento: {target_width}x{target_height}")

    # Configurar límites por defecto
    if max_size is None:
        max_size = (4000, 4000)  # Límite razonable para evitar memoria excesiva
    if min_size is None:
        min_size = (patch_size, patch_size)  # Al menos un patch

    if method == 'maintain_scale':
        new_width, new_height, scale_factor = _maintain_scale_resize(
            original_size, target_base_size, max_size, min_size, patch_size
        )

    elif method == 'fit_to_multiple':
        new_width, new_height, scale_factor = _fit_to_multiple_resize(
            original_size, patch_size, max_size, min_size
        )

    elif method == 'preserve_aspect':
        new_width, new_height, scale_factor = _preserve_aspect_resize(
            original_size, target_base_size, max_size, min_size, patch_size
        )

    elif method == 'adaptive':
        new_width, new_height, scale_factor = _adaptive_resize(
            original_size, target_base_size, max_size, min_size, patch_size
        )

    else:
        raise ValueError(f"Método '{method}' no reconocido")

    # Asegurar que las dimensiones sean múltiplos del patch_size
    new_width = _round_to_multiple(new_width, patch_size)
    new_height = _round_to_multiple(new_height, patch_size)

    # Aplicar redimensionamiento
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    final_size = (new_width, new_height)
    actual_scale_factor = new_width / original_width  # Recalcular escala real

    print(f"Imagen redimensionada: {new_width}x{new_height}")
    print(f"Factor de escala aplicado: {actual_scale_factor:.3f}")
    print(f"Patches resultantes: {new_width//patch_size}x{new_height//patch_size} = {(new_width//patch_size)*(new_height//patch_size)} patches")

    return resized_image, actual_scale_factor, final_size

def _maintain_scale_resize(original_size, target_base_size, max_size, min_size, patch_size):
    """Mantiene la escala de elementos similar al entrenamiento"""
    orig_w, orig_h = original_size
    target_w, target_h = target_base_size
    max_w, max_h = max_size
    min_w, min_h = min_size

    # Calcular qué tan grande es la imagen comparada con el tamaño de entrenamiento
    scale_w = orig_w / target_w
    scale_h = orig_h / target_h

    # Usar la escala promedio para mantener proporciones similares
    avg_scale = (scale_w + scale_h) / 2

    # Si la imagen es mucho más grande, reducirla para mantener densidad de información
    if avg_scale > 2.0:  # Imagen mucho más grande
        reduction_factor = math.sqrt(avg_scale / 1.5)  # Reducir pero no tanto
        new_width = int(orig_w / reduction_factor)
        new_height = int(orig_h / reduction_factor)
    elif avg_scale < 0.5:  # Imagen mucho más pequeña
        enlargement_factor = math.sqrt(2.0 / avg_scale)
        new_width = int(orig_w * enlargement_factor)
        new_height = int(orig_h * enlargement_factor)
    else:
        # Tamaño razonable, mantener original
        new_width, new_height = orig_w, orig_h

    # Aplicar límites
    new_width = max(min_w, min(max_w, new_width))
    new_height = max(min_h, min(max_h, new_height))

    scale_factor = new_width / orig_w
    return new_width, new_height, scale_factor

def _fit_to_multiple_resize(original_size, patch_size, max_size, min_size):
    """Ajusta a múltiplos exactos del patch_size manteniendo aspect ratio"""
    orig_w, orig_h = original_size
    max_w, max_h = max_size
    min_w, min_h = min_size

    # Calcular número óptimo de patches manteniendo aspect ratio
    aspect_ratio = orig_w / orig_h

    # Estimar número de patches basado en un tamaño razonable
    target_patches_total = max(4, min(100, (orig_w * orig_h) // (patch_size * patch_size * 4)))

    # Calcular dimensiones en patches
    patches_h = int(math.sqrt(target_patches_total / aspect_ratio))
    patches_w = int(patches_h * aspect_ratio)

    # Asegurar al menos 1 patch en cada dimensión
    patches_w = max(1, patches_w)
    patches_h = max(1, patches_h)

    new_width = patches_w * patch_size
    new_height = patches_h * patch_size

    # Aplicar límites
    new_width = max(min_w, min(max_w, new_width))
    new_height = max(min_h, min(max_h, new_height))

    scale_factor = new_width / orig_w
    return new_width, new_height, scale_factor

def _preserve_aspect_resize(original_size, target_base_size, max_size, min_size, patch_size):
    """Preserva aspect ratio ajustando a un tamaño manejable"""
    orig_w, orig_h = original_size
    target_w, target_h = target_base_size
    max_w, max_h = max_size
    min_w, min_h = min_size

    # Calcular escalas para ajustar al tamaño objetivo
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h

    # Usar la escala menor para mantener aspect ratio
    scale = min(scale_w, scale_h)

    # Si resulta demasiado pequeño, usar escala mayor
    if scale < 0.3:
        scale = max(scale_w, scale_h) * 0.7

    new_width = int(orig_w * scale)
    new_height = int(orig_h * scale)

    # Aplicar límites
    new_width = max(min_w, min(max_w, new_width))
    new_height = max(min_h, min(max_h, new_height))

    scale_factor = new_width / orig_w
    return new_width, new_height, scale_factor

def _adaptive_resize(original_size, target_base_size, max_size, min_size, patch_size):
    """Estrategia adaptativa que combina métodos según el caso"""
    orig_w, orig_h = original_size
    target_w, target_h = target_base_size

    # Determinar qué estrategia usar según el tamaño
    size_ratio = (orig_w * orig_h) / (target_w * target_h)

    if size_ratio > 4.0:  # Imagen muy grande
        return _maintain_scale_resize(original_size, target_base_size, max_size, min_size, patch_size)
    elif size_ratio < 0.25:  # Imagen muy pequeña
        return _preserve_aspect_resize(original_size, target_base_size, max_size, min_size, patch_size)
    else:  # Tamaño intermedio
        return _fit_to_multiple_resize(original_size, patch_size, max_size, min_size)

def _round_to_multiple(value, multiple):
    """Redondea un valor al múltiplo más cercano"""
    return int(round(value / multiple) * multiple)

def analyze_resize_options(image, target_base_size=(1280, 720), patch_size=224):
    """
    Analiza diferentes opciones de redimensionamiento para una imagen.
    Útil para decidir qué método usar.
    """
    # image = cv2.imread(image_path)
    if image is None:
         raise ValueError(f"No se pudo cargar la imagen: {image}")

    original_height, original_width = image.shape[:2]

    print("="*60)
    print("ANÁLISIS DE OPCIONES DE REDIMENSIONAMIENTO")
    print("="*60)
    print(f"Imagen original: {original_width}x{original_height}")
    print(f"Tamaño base entrenamiento: {target_base_size[0]}x{target_base_size[1]}")
    print(f"Tamaño del patch: {patch_size}x{patch_size}")
    print("-"*60)

    methods = ['maintain_scale', 'fit_to_multiple', 'preserve_aspect', 'adaptive']

    for method in methods:
        try:
            _, scale_factor, final_size = smart_resize_for_patches(
                image.copy(), target_base_size, patch_size, method
            )
            patches_w = final_size[0] // patch_size
            patches_h = final_size[1] // patch_size
            total_patches = patches_w * patches_h

            print(f"\nMétodo '{method}':")
            print(f"  Tamaño final: {final_size[0]}x{final_size[1]}")
            print(f"  Factor escala: {scale_factor:.3f}")
            print(f"  Patches: {patches_w}x{patches_h} = {total_patches} total")
            print(f"  Cambio de área: {(scale_factor**2)*100:.1f}%")

        except Exception as e:
            print(f"\nMétodo '{method}': ERROR - {str(e)}")

    print("="*60)

# Función de conveniencia para uso fácil
def prepare_image_for_analysis(image, patch_size=224, strategy='adaptive'):
    """
    Función simple para preparar una imagen para análisis de patches.

    Args:
        image_path: Ruta a la imagen
        patch_size: Tamaño del patch (default: 224)
        strategy: Estrategia de redimensionamiento (default: 'adaptive')

    Returns:
        resized_image: Imagen lista para análisis
        info: Diccionario con información del redimensionamiento
    """
    # image = cv2.imread(image_path)
    if image is None:
         raise ValueError(f"No se pudo cargar la imagen: {image}")

    resized_image, scale_factor, final_size = smart_resize_for_patches(
        image, method=strategy, patch_size=patch_size
    )

    info = {
        'original_size': (image.shape[1], image.shape[0]),
        'final_size': final_size,
        'scale_factor': scale_factor,
        'total_patches': (final_size[0] // patch_size) * (final_size[1] // patch_size),
        'patches_grid': (final_size[0] // patch_size, final_size[1] // patch_size)
    }

    return resized_image, info

# image_path = "/content/drive/MyDrive/Proyecto IA3/Planos/fachada_reconstruida3.jpg"

#     # Ver todas las opciones disponibles
# analyze_resize_options(image_path)

#     # Usar la función simple
# resized_image, info = prepare_image_for_analysis(image_path, strategy='adaptive')
# print(f"\nImagen preparada:")
# print(f"Tamaño final: {info['final_size']}")
# print(f"Total patches: {info['total_patches']}")
# print(f"Grid de patches: {info['patches_grid']}")