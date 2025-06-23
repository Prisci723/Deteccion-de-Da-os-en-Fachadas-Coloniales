String getImageTitle(String key) {
  switch (key) {
    case 'damage_detection':
      return 'Detección de Daños';
    case 'damage_segmentation':
      return 'Segmentación de Daños';
    default:
      return key.replaceAll('_', ' ').toUpperCase();
  }
}

String getImageDescription(String key) {
  switch (key) {
    case 'damage_detection':
      return 'Patches marcados con nivel de confianza de detección de daños';
    case 'damage_segmentation':
      return 'Segmentación por tipos de daños con colores diferenciados';
    default:
      return 'Imagen de análisis';
  }
}
