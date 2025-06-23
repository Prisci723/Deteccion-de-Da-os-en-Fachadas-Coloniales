
import numpy as np
DAMAGE_TYPES = ['Fondo', 'deformacion', 'desprendimiento', 'deterioro','ensanchamiento','filtracion', 'fisuracion', 'grietas', 'humedad', 'humedad_interna', 'hundimiento']
ARCHITECTURAL_ELEMENTS = ['abertura', 'base_muro', 'espadana', 'muro', 'techo']
DAMAGE_DETECTION_SIZE = 224
DAMAGE_SEGMENTATION_SIZE = 448
ARCHITECTURAL_ELEMENT_SIZE = 228
def generate_damage_analysis_improved(damage_mask, element_mask, return_format='langchain'):
    """
    Versión mejorada de tu función de análisis de daños
    """
    stats = {}
    total_pixels = damage_mask.size
    total_damage_pixels = np.sum(damage_mask > 0)
    
    # Análisis general
    general_stats = {
        'total_area': total_pixels,
        'total_damage_area': total_damage_pixels,
        'damage_percentage': (total_damage_pixels / total_pixels) * 100 if total_pixels > 0 else 0,
        'damage_types_found': []
    }
    
    # Análisis por tipo de daño
    damage_type_stats = {}
    for damage_id in range(1, len(DAMAGE_TYPES)):
        damage_pixels = np.sum(damage_mask == damage_id)
        if damage_pixels > 0:
            damage_type_stats[DAMAGE_TYPES[damage_id]] = {
                'area': damage_pixels,
                'percentage_of_total': (damage_pixels / total_pixels) * 100,
                'percentage_of_damage': (damage_pixels / total_damage_pixels) * 100 if total_damage_pixels > 0 else 0
            }
            general_stats['damage_types_found'].append(DAMAGE_TYPES[damage_id])

    # Análisis por elemento arquitectónico
    element_stats = {}
    for elem_id in range(1, len(ARCHITECTURAL_ELEMENTS)):
        elem_pixels = element_mask == elem_id
        elem_area = np.sum(elem_pixels)

        if elem_area > 0:
            elem_damage_pixels = np.sum((damage_mask > 0) & elem_pixels)
            elem_stat = {
                'nombre': ARCHITECTURAL_ELEMENTS[elem_id],
                'area_total': elem_area,
                'area_dañada': elem_damage_pixels,
                'porcentaje_daño': (elem_damage_pixels / elem_area) * 100 if elem_area > 0 else 0,
                'daños_por_tipo': {}
            }

            # Análisis de daños por tipo en este elemento
            for damage_id in range(1, len(DAMAGE_TYPES)):
                damage_in_element = np.sum((damage_mask == damage_id) & elem_pixels)
                if damage_in_element > 0:
                    elem_stat['daños_por_tipo'][DAMAGE_TYPES[damage_id]] = {
                        'area': damage_in_element,
                        'porcentaje_elemento': (damage_in_element / elem_area) * 100
                    }

            element_stats[ARCHITECTURAL_ELEMENTS[elem_id]] = elem_stat

    # Combinarlo todo
    stats = {
        'resumen_general': general_stats,
        'analisis_por_tipo_daño': damage_type_stats,
        'analisis_por_elemento': element_stats,
        'recomendaciones': generate_recommendations_improved(damage_type_stats, element_stats)
    }

    if return_format == 'dict':
        return stats
    else:
        return format_for_langchain_improved(stats)

def generate_recommendations_improved(damage_type_stats, element_stats):
    """
    Genera recomendaciones específicas basadas en tus tipos de daños
    """
    recommendations = []
    
    # Recomendaciones basadas en tipos de daño específicos
    for damage_type, stats in damage_type_stats.items():
        if stats['percentage_of_total'] > 5:  # Si el daño ocupa más del 5% del área total
            if damage_type == 'fisuracion':
                recommendations.append("URGENTE: Se detectó fisuración extensa. Evaluar estabilidad estructural inmediatamente.")
            elif damage_type == 'grietas':
                recommendations.append("CRÍTICO: Presencia de grietas. Inspección estructural requerida.")
            elif damage_type == 'humedad' or damage_type == 'humedad_interna':
                recommendations.append("Revisar sistemas de impermeabilización y drenaje por problemas de humedad.")
            elif damage_type == 'filtracion':
                recommendations.append("Inspeccionar y reparar sistemas de sellado para prevenir filtraciones.")
            elif damage_type == 'desprendimiento':
                recommendations.append("SEGURIDAD: Riesgo de desprendimientos. Evaluar áreas de tránsito peatonal.")
            elif damage_type == 'deterioro':
                recommendations.append("Planificar mantenimiento preventivo para evitar deterioro progresivo.")
            elif damage_type == 'hundimiento':
                recommendations.append("CRÍTICO: Hundimiento detectado. Evaluación geotécnica necesaria.")
            elif damage_type == 'deformacion':
                recommendations.append("Evaluar causas de deformación estructural y tomar medidas correctivas.")
            elif damage_type == 'ensanchamiento':
                recommendations.append("Monitorear evolución del ensanchamiento y evaluar refuerzo estructural.")
    
    # Recomendaciones basadas en elementos arquitectónicos afectados
    for element_name, stats in element_stats.items():
        if stats['porcentaje_daño'] > 10:  # Si el elemento tiene más del 10% de daño
            if element_name == 'muro':
                recommendations.append(f"Inspección detallada del muro requerida ({stats['porcentaje_daño']:.1f}% dañado).")
            elif element_name == 'techo':
                recommendations.append(f"Revisión urgente del techo por riesgo de infiltraciones ({stats['porcentaje_daño']:.1f}% dañado).")
            elif element_name == 'abertura':
                recommendations.append(f"Evaluar marcos y sellos de aberturas ({stats['porcentaje_daño']:.1f}% dañado).")
            elif element_name == 'base_muro':
                recommendations.append(f"Inspeccionar cimentación y base del muro ({stats['porcentaje_daño']:.1f}% dañado).")
            elif element_name == 'espadana':
                recommendations.append(f"Revisar estabilidad de espadaña ({stats['porcentaje_daño']:.1f}% dañado).")
    
    # Recomendación general si no hay daños críticos
    if not recommendations:
        recommendations.append("Continuar con programa de mantenimiento preventivo regular.")
    
    return recommendations

def format_for_langchain_improved(stats):
    """
    Formatea los resultados para LangChain con tu información específica
    """
    text = "=== ANÁLISIS DE DAÑOS EN FACHADA ===\n\n"
    
    # Resumen general
    general = stats['resumen_general']
    text += f"RESUMEN GENERAL:\n"
    #text += f"- Área total analizada: {general['total_area']:,} píxeles\n"
    #text += f"- Área con daños: {general['total_damage_area']:,} píxeles\n"
    text += f"- Porcentaje de daño: {general['damage_percentage']:.2f}%\n"
    text += f"- Tipos de daños encontrados: {', '.join(general['damage_types_found'])}\n\n"
    
    # Análisis por tipo de daño
    if stats['analisis_por_tipo_daño']:
        text += "ANÁLISIS POR TIPO DE DAÑO:\n"
        for damage_type, data in stats['analisis_por_tipo_daño'].items():
            text += f"- {damage_type.upper()}:\n"
            #text += f"  * Área afectada: {data['area']:,} píxeles\n"
            text += f"  * % del área total: {data['percentage_of_total']:.2f}%\n"
            text += f"  * % del daño total: {data['percentage_of_damage']:.2f}%\n"
        text += "\n"
    
    # Análisis por elemento arquitectónico
    if stats['analisis_por_elemento']:
        text += "ANÁLISIS POR ELEMENTO ARQUITECTÓNICO:\n"
        for element_name, data in stats['analisis_por_elemento'].items():
            text += f"- {element_name.upper().replace('_', ' ')}:\n"
            #text += f"  * Área total: {data['area_total']:,} píxeles\n"
            #text += f"  * Área dañada: {data['area_dañada']:,} píxeles\n"
            text += f"  * Porcentaje de daño: {data['porcentaje_daño']:.2f}%\n"
            
            if data['daños_por_tipo']:
                text += f"  * Tipos de daños presentes:\n"
                for damage_type, damage_data in data['daños_por_tipo'].items():
                    text += f"    - {damage_type}: {damage_data['porcentaje_elemento']:.2f}% del elemento\n"
            text += "\n"
    
    # Recomendaciones
    if stats['recomendaciones']:
        text += "RECOMENDACIONES:\n"
        for i, rec in enumerate(stats['recomendaciones'], 1):
            text += f"{i}. {rec}\n"
    
    return text
