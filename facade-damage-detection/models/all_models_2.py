import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from models.langchain_results import generate_damage_analysis_improved
from models.model_utils import get_transforms
from models.visualize_results import visualize_results_improved
from PIL import Image

# Tus constantes originales
DAMAGE_TYPES = ['Fondo', 'deformacion', 'desprendimiento', 'deterioro','ensanchamiento','filtracion', 'fisuracion', 'grietas', 'humedad', 'humedad_interna', 'hundimiento']
ARCHITECTURAL_ELEMENTS = ['abertura', 'base_muro', 'espadana', 'muro', 'techo']
DAMAGE_DETECTION_SIZE = 224
DAMAGE_SEGMENTATION_SIZE = 448
ARCHITECTURAL_ELEMENT_SIZE = 228

# Configurar logging
logger = logging.getLogger(__name__)

@dataclass
class ImprovedConfig:
    """Configuración mejorada para tu sistema específico"""
    # Thresholds de confianza
    damage_detection_threshold: float = 0.75
    damage_segmentation_threshold: float = 0.65
    architectural_elements_threshold: float = 0.6
    
    # Configuración de procesamiento
    batch_size: int = 16
    device: str = 'auto'
    
    # Parámetros de patches - usando tus valores originales
    damage_patch_size: int = DAMAGE_DETECTION_SIZE
    segmentation_patch_size: int = DAMAGE_SEGMENTATION_SIZE
    arch_patch_size: int = ARCHITECTURAL_ELEMENT_SIZE
    overlap_ratio: float = 0.50  # 25% de solapamiento (tu valor original de 56/224)
    
    # Configuración de salida
    save_intermediate_results: bool = True
    visualization_dpi: int = 300

class ImprovedFacadeAnalyzer:
    """Analizador mejorado adaptado a tu implementación específica"""
    
    def __init__(self, models=None, transforms=None, config=None):
        # Hacer los parámetros opcionales para evitar errores de inicialización
        if models is None or transforms is None:
            raise ValueError("Se requieren models y transforms para inicializar ImprovedFacadeAnalyzer")
            
        self.damage_detection, self.damage_segmentation, self.architectural_elements = models
        self.damage_detection_transform, self.damage_segmentation_transform, self.architectural_elements_transform = transforms
        self.config = config or ImprovedConfig()
        
        # Configurar dispositivo
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"Analizador mejorado inicializado en dispositivo: {self.device}")
    
    def expand_patch_coordinates(self, original_coords: Dict, target_size: int, image_shape: Tuple[int, int]) -> Dict:
        """
        Expande las coordenadas de un patch para obtener un patch más grande centrado.
        
        Args:
            original_coords: Diccionario con coordenadas del patch original {'x', 'y', 'width', 'height'}
            target_size: Tamaño objetivo del patch expandido (ej: 448)
            image_shape: Tupla con (height, width) de la imagen completa
        
        Returns:
            Dict: Nuevas coordenadas expandidas y información del ajuste
        """
        img_height, img_width = image_shape
        
        # Coordenadas del centro del patch original
        center_x = original_coords['x'] + original_coords['width'] // 2
        center_y = original_coords['y'] + original_coords['height'] // 2
        
        # Calcular las nuevas coordenadas para el patch expandido
        half_target = target_size // 2
        new_x = center_x - half_target
        new_y = center_y - half_target
        
        # Ajustar si se sale de los límites de la imagen
        padding_left = max(0, -new_x)
        padding_top = max(0, -new_y)
        padding_right = max(0, (new_x + target_size) - img_width)
        padding_bottom = max(0, (new_y + target_size) - img_height)
        
        # Ajustar coordenadas finales
        final_x = max(0, new_x)
        final_y = max(0, new_y)
        final_width = min(target_size - padding_right, img_width - final_x)
        final_height = min(target_size - padding_bottom, img_height - final_y)
        
        return {
            'x': final_x,
            'y': final_y,
            'width': final_width,
            'height': final_height,
            'target_size': target_size,
            'padding': {
                'left': padding_left,
                'top': padding_top,
                'right': padding_right,
                'bottom': padding_bottom
            },
            'center_offset': {
                'x': center_x - original_coords['x'],
                'y': center_y - original_coords['y']
            }
        }
    
    def extract_expanded_patch(self, image: np.ndarray, original_coords: Dict, target_size: int) -> Tuple[np.ndarray, Dict]:
        """
        Extrae un patch expandido de la imagen centrado en el patch original.
        
        Args:
            image: Imagen completa (numpy array)
            original_coords: Coordenadas del patch original donde se detectó daño
            target_size: Tamaño objetivo del patch (ej: 448)
        
        Returns:
            Tuple: (patch_expandido, info_expansion)
        """
        img_height, img_width = image.shape[:2]
        
        # Calcular coordenadas expandidas
        expanded_coords = self.expand_patch_coordinates(original_coords, target_size, (img_height, img_width))
        
        # Extraer el patch de la imagen
        patch = image[expanded_coords['y']:expanded_coords['y'] + expanded_coords['height'],
                     expanded_coords['x']:expanded_coords['x'] + expanded_coords['width']]
        
        # Si el patch no tiene el tamaño exacto debido a bordes, añadir padding
        padding = expanded_coords['padding']
        if any(padding.values()):
            patch = cv2.copyMakeBorder(
                patch,
                padding['top'],
                padding['bottom'], 
                padding['left'],
                padding['right'],
                cv2.BORDER_REFLECT_101  # O cv2.BORDER_CONSTANT con value=0
            )
        
        return patch, expanded_coords
    
    def generate_patches_improved(self, image: Union[np.ndarray, Image.Image], patch_size: int) -> List[Dict]:
        """
        Genera patches mejorados manteniendo tu lógica original
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        height, width = image.shape[:2]
        overlap = int(patch_size * self.config.overlap_ratio)
        stride = patch_size - overlap
        
        patches = []
        patch_idx = 0
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = image[y:y + patch_size, x:x + patch_size]
                
                patches.append({
                    'patch': patch,
                    'coords': {
                        'x': x,
                        'y': y,
                        'width': patch_size,
                        'height': patch_size,
                        'index': patch_idx
                    },
                    'index': patch_idx,
                    'location': (x, y)
                })
                patch_idx += 1
        
        return patches
    
    def detect_damage_improved(self, patches: List[Dict]) -> List[Dict]:
        """
        Detección de daños mejorada con procesamiento por lotes
        IMPORTANTE: Mantengo tu lógica donde clase 0 = damage, clase 1 = no damage
        """
        damage_patches = []
        
        # Procesar en lotes
        for i in range(0, len(patches), self.config.batch_size):
            batch_patches = patches[i:i + self.config.batch_size]
            batch_tensors = []
            
            # Preparar batch
            for patch_info in batch_patches:
                patch_pil = Image.fromarray(patch_info['patch'])
                patch_tensor = self.damage_detection_transform(patch_pil)
                batch_tensors.append(patch_tensor)
            
            # Convertir a tensor batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Predecir
            with torch.no_grad():
                outputs = self.damage_detection(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Procesar resultados del batch
                for j, patch_info in enumerate(batch_patches):
                    predicted_class = predictions[j].item()
                    
                    # En tu sistema: clase 0 = damage, clase 1 = no damage
                    if predicted_class == 0:
                        damage_prob = probabilities[j, 0].item()  # Probabilidad de daño (clase 0)
                        
                        if damage_prob >= self.config.damage_detection_threshold:
                            patch_result = patch_info.copy()
                            patch_result.update({
                                'damage_probability': damage_prob,
                                'confidence': damage_prob,
                                'class_name': 'damage',
                                'predicted_class': predicted_class
                            })
                            damage_patches.append(patch_result)
        
        logger.info(f"Detectados {len(damage_patches)} patches con daños de {len(patches)} totales")
        return damage_patches
    
    def segment_damage_types_improved(self, damage_patches: List[Dict], original_image: np.ndarray) -> List[Dict]:
        """
        Segmentación mejorada que extrae patches de 448x448 centrados en los daños detectados de 224x224
        
        Args:
            damage_patches: Lista de patches donde se detectaron daños (224x224)
            original_image: Imagen completa original para extraer patches expandidos
        """
        if not self.damage_segmentation:
            logger.warning("Modelo de segmentación no disponible")
            return damage_patches
        
        segmentation_results = []
        
        for patch_info in tqdm(damage_patches, desc="Segmentando tipos de daños con patches expandidos"):
            try:
                # Extraer patch expandido de 448x448 centrado en el daño detectado
                expanded_patch, expansion_info = self.extract_expanded_patch(
                    original_image, 
                    patch_info['coords'], 
                    self.config.segmentation_patch_size
                )
                
                # Verificar que el patch tenga el tamaño correcto
                if expanded_patch.shape[:2] != (self.config.segmentation_patch_size, self.config.segmentation_patch_size):
                    logger.warning(f"Patch expandido tiene tamaño incorrecto: {expanded_patch.shape[:2]}")
                    continue
                
                # Procesar con el modelo de segmentación
                patch_pil = Image.fromarray(expanded_patch)
                patch_tensor = self.damage_segmentation_transform(patch_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.damage_segmentation(patch_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
                    
                    # Calcular confianza promedio para áreas segmentadas
                    max_probs = torch.max(probabilities, dim=1)[0].squeeze().cpu().numpy()
                    segmentation_confidence = np.mean(max_probs[predicted_mask > 0]) if np.any(predicted_mask > 0) else 0.0
                    
                    if segmentation_confidence >= self.config.damage_segmentation_threshold:
                        result = patch_info.copy()
                        result.update({
                            'segmentation_mask': predicted_mask,
                            'segmentation_confidence': segmentation_confidence,
                            'damage_types_detected': np.unique(predicted_mask[predicted_mask > 0]).tolist(),
                            'damage_types_names': [DAMAGE_TYPES[i] for i in np.unique(predicted_mask[predicted_mask > 0]).tolist()],
                            'expanded_coords': expansion_info,  # Información sobre la expansión
                            'expanded_patch_size': self.config.segmentation_patch_size
                        })
                        segmentation_results.append(result)
                    
            except Exception as e:
                logger.error(f"Error segmentando patch {patch_info['index']}: {str(e)}")
                continue
        
        logger.info(f"Segmentación completada para {len(segmentation_results)} patches")
        return segmentation_results
    
    def classify_architectural_elements_improved(self, damage_patches: List[Dict]) -> List[Dict]:
        """
        Clasificación de elementos arquitectónicos SOLO en patches con daños
        """
        if not self.architectural_elements:
            logger.warning("Modelo de elementos arquitectónicos no disponible")
            return damage_patches
        
        architectural_results = []
        
        for patch_info in tqdm(damage_patches, desc="Clasificando elementos arquitectónicos"):
            try:
                # Redimensionar patch al tamaño de arquitectura (228x228)
                patch_resized = cv2.resize(patch_info['patch'], 
                                         (self.config.arch_patch_size, self.config.arch_patch_size))
                patch_pil = Image.fromarray(patch_resized)
                patch_tensor = self.architectural_elements_transform(patch_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.architectural_elements(patch_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    max_prob, predicted_class = torch.max(probabilities, 1)
                    
                    element_confidence = max_prob.item()
                    element_class = predicted_class.item()
                    
                    if element_confidence >= self.config.architectural_elements_threshold and element_class < len(ARCHITECTURAL_ELEMENTS):
                        result = patch_info.copy()
                        result.update({
                            'architectural_element': element_class,
                            'architectural_confidence': element_confidence,
                            'architectural_element_name': ARCHITECTURAL_ELEMENTS[element_class]
                        })
                        architectural_results.append(result)
                    
            except Exception as e:
                logger.error(f"Error clasificando elemento en patch {patch_info['index']}: {str(e)}")
                continue
        
        logger.info(f"Elementos arquitectónicos clasificados en {len(architectural_results)} patches")
        return architectural_results

# Función de conveniencia para crear el analizador
def create_analyzer(models, transforms, config=None):
    """
    Función de conveniencia para crear un ImprovedFacadeAnalyzer
    
    Args:
        models: Tupla con (damage_detection, damage_segmentation, architectural_elements)
        transforms: Tupla con las transformaciones correspondientes
        config: Configuración opcional (ImprovedConfig)
    
    Returns:
        ImprovedFacadeAnalyzer: Instancia configurada del analizador
    """
    if config is None:
        config = ImprovedConfig()
    
    return ImprovedFacadeAnalyzer(models, transforms, config)

def improved_load_and_analyze_facade(facade_img, models, transforms, 
                                   patch_size=224, stride=112, config=None):
    """
    Versión mejorada de tu función original load_and_analyze_facade
    Mantiene la compatibilidad pero añade las mejoras
    """
    if config is None:
        config = ImprovedConfig()
    
    # Crear el analizador con los parámetros correctos
    analyzer = create_analyzer(models, transforms, config)
    damage_detection, damage_segmentation, architectural_elements = models
    
    device = analyzer.device
    
    # Cargar la imagen de la fachada
    if facade_img is None:
        raise FileNotFoundError("No se pudo cargar la imagen")

    facade_img = cv2.cvtColor(facade_img, cv2.COLOR_BGR2RGB)
    facade_height, facade_width = facade_img.shape[:2]

    # Crear una copia para visualización
    facade_visualization = facade_img.copy()
    damage_mask = np.zeros((facade_height, facade_width), dtype=np.uint8)
    element_mask = np.zeros((facade_height, facade_width), dtype=np.uint8)

    # Generar patches con mejoras
    print("Generando patches mejorados...")
    patches_info = analyzer.generate_patches_improved(facade_img, patch_size)
    
    # 1. Detectar daños en los patches (con procesamiento por lotes)
    print("Analizando patches para detección de daños con batch processing...")
    damage_patches_info = analyzer.detect_damage_improved(patches_info)
    
    # Extraer información para compatibilidad
    damage_patches = []
    damage_locations = []
    
    for patch_info in damage_patches_info:
        damage_patches.append(patch_info['patch'])
        damage_locations.append(patch_info['location'])
        
        # Dibujar rectángulo para visualización
        coords = patch_info['coords']
        confidence = patch_info['confidence']
        color_intensity = int(255 * confidence)
        cv2.rectangle(facade_visualization, 
                     (coords['x'], coords['y']), 
                     (coords['x'] + coords['width'], coords['y'] + coords['height']),
                     (color_intensity, 0, 0), 2)
        
        # Añadir texto con confianza
        cv2.putText(facade_visualization, f"{confidence:.2f}", 
                   (coords['x'], coords['y'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (color_intensity, 0, 0), 1)

    print(f"Se encontraron {len(damage_patches)} patches con daños de un total de {len(patches_info)} patches.")

    # 2. Segmentar los daños en los patches con daños (MEJORADO - con patches expandidos)
    if len(damage_patches_info) > 0 and damage_segmentation is not None:
        print("Segmentando tipos de daños con patches expandidos de 448x448...")
        segmentation_results = analyzer.segment_damage_types_improved(damage_patches_info, facade_img)
        
        # Actualizar la máscara global usando la información de expansión
        for result in segmentation_results:
            expanded_coords = result['expanded_coords']
            mask = result['segmentation_mask']
            
            x, y = expanded_coords['x'], expanded_coords['y']
            mask_height, mask_width = mask.shape
            
            # Asegurar que no nos salgamos de los límites
            end_y = min(y + mask_height, facade_height)
            end_x = min(x + mask_width, facade_width)
            actual_height = end_y - y
            actual_width = end_x - x
            
            # Actualizar la máscara global
            for class_id in range(1, len(DAMAGE_TYPES)):
                class_mask = (mask[:actual_height, :actual_width] == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0:
                    damage_mask[y:end_y, x:end_x][class_mask > 0] = class_id

    # 3. Clasificar elementos arquitectónicos (solo en patches con daños)
    if len(damage_patches_info) > 0 and architectural_elements is not None:
        print("Clasificando elementos arquitectónicos en áreas dañadas...")
        architectural_results = analyzer.classify_architectural_elements_improved(damage_patches_info)
        
        # Actualizar la máscara de elementos
        for result in architectural_results:
            coords = result['coords']
            element_id = result['architectural_element']
            
            x, y = coords['x'], coords['y']
            patch_size_actual = coords['width']
            
            if element_id > 0 and element_id < len(ARCHITECTURAL_ELEMENTS):
                element_mask[y:y+patch_size_actual, x:x+patch_size_actual] = element_id

    return facade_img, facade_visualization, damage_mask, element_mask, damage_patches, damage_locations

# Resto de funciones mantienen la misma estructura...
def improved_process_plano(plano_path, damage_classifier, damage_segmenter, arch_classifier, 
                          output_dir='./output', config=None):
    """
    Versión mejorada de tu función original process_plano
    """
    if config is None:
        config = ImprovedConfig()
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar imagen del plano
    try:
        plano = Image.open(plano_path)
        logger.info(f"Plano cargado: {plano_path}, tamaño: {plano.size}")
    except Exception as e:
        logger.error(f"Error al cargar el plano: {str(e)}")
        return None
    
    # Preparar modelos y transforms (asumiendo que tienes get_transforms())
    models = (damage_classifier, damage_segmenter, arch_classifier)
    transforms = get_transforms()  # Tu función existente
    
    # Usar la función mejorada con la función de conveniencia
    facade_img, facade_visualization, damage_mask, element_mask, damage_patches, damage_locations = \
        improved_load_and_analyze_facade(np.array(plano), models, transforms, config=config)
    
    # Generar análisis mejorado
    damage_analysis = generate_damage_analysis_improved(damage_mask, element_mask)
    
    # Visualizar resultados mejorados
    logger.info("Generando visualización de resultados...")
    visualization = visualize_results_improved(facade_img, facade_visualization, damage_mask, element_mask)
    
    # Guardar visualización
    output_path = os.path.join(output_dir, "improved_result_visualization.png")
    visualization.save(output_path)
    logger.info(f"Visualización guardada en: {output_path}")
    
    return {
        "damage_patches": len(damage_patches),
        "damage_locations": damage_locations,
        "damage_analysis": damage_analysis,
        "damage_mask": damage_mask,
        "element_mask": element_mask,
        "visualization": visualization,
        "config_used": config
    }

def analyze_facade_simple(facade_img, models, transforms, config_preset='production'):
    """
    Función simplificada para análisis de fachada
    
    Args:
        facade_img: Imagen de la fachada (numpy array, PIL Image, o path)
        models: Tupla con (damage_detection, damage_segmentation, architectural_elements)
        transforms: Tupla con las transformaciones
        config_preset: 'production', 'research', 'fast', o un objeto ImprovedConfig
    
    Returns:
        Dict con resultados completos del análisis
    """
    # Configurar según preset
    if isinstance(config_preset, str):
        config = get_config_preset(config_preset)
    else:
        config = config_preset or ImprovedConfig()
    
    # Cargar imagen si es un path
    if isinstance(facade_img, str):
        facade_img = cv2.imread(facade_img)
        facade_img = cv2.cvtColor(facade_img, cv2.COLOR_BGR2RGB)
    elif isinstance(facade_img, Image.Image):
        facade_img = np.array(facade_img)
    
    # Análisis completo
    facade_img_processed, facade_visualization, damage_mask, element_mask, damage_patches, damage_locations = \
        improved_load_and_analyze_facade(facade_img, models, transforms, config=config)
    
    # Generar análisis
    damage_analysis = generate_damage_analysis_improved(damage_mask, element_mask)
    
    # Crear visualización
    visualization = visualize_results_improved(facade_img_processed, facade_visualization, damage_mask, element_mask)
    
    return {
        'facade_img': facade_img_processed,
        'facade_visualization': facade_visualization,
        'damage_mask': damage_mask,
        'element_mask': element_mask,
        'damage_patches': damage_patches,
        'damage_locations': damage_locations,
        'damage_analysis': damage_analysis,
        'visualization': visualization
    }

# Función para obtener configuraciones preestablecidas
def get_config_preset(preset_name: str) -> ImprovedConfig:
    """
    Obtiene configuraciones preestablecidas
    """
    presets = {
        'production': ImprovedConfig(
            damage_detection_threshold=0.5,
            damage_segmentation_threshold=0.4,
            architectural_elements_threshold=0.4,
            batch_size=32,
            save_intermediate_results=False,
            visualization_dpi=150
        ),
        'research': ImprovedConfig(
            damage_detection_threshold=0.5,
            damage_segmentation_threshold=0.4,
            architectural_elements_threshold=0.4,
            batch_size=16,
            save_intermediate_results=True,
            visualization_dpi=300
        ),
        'fast': ImprovedConfig(
            damage_detection_threshold=0.5,
            damage_segmentation_threshold=0.4,
            architectural_elements_threshold=0.4,
            batch_size=64,
            overlap_ratio=0.15,
            save_intermediate_results=False,
            visualization_dpi=100
        )
    }
    
    return presets.get(preset_name, ImprovedConfig())