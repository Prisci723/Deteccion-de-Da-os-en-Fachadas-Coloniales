from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import os
import cv2

from models.model_utils import load_test_models
from models.all_models_2 import ImprovedConfig, generate_damage_analysis_improved, improved_load_and_analyze_facade, get_transforms
from models.visualize_results import compress_and_encode_images, create_individual_analysis_images
from utils.image_processing import base64_to_cv2, get_adaptive_config_for_panorama, base64_to_rgb, cv2_to_base64
from utils.resize import prepare_image_for_analysis
from langchain_integration.init_langchain import DiagnosisSystem
from sfm.proyectosfmvariasimagenes import (
    analizar_danos,
    detectar_caracteristicas,
    encontrar_correspondencias,
    estimar_homografias,
    componer_homografias,
    calcular_limites_panorama,
    crear_mosaico
)

app = FastAPI(title="Facade Analysis API", version="1.0.0")

# Configurar CORS para permitir requests desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constantes
DAMAGE_DETECTION_SIZE = 224

# Modelos globales (se cargan una vez al iniciar)
models = None
transforms_list = None
diagnosis_system = None

# Modelos Pydantic para requests
class ImageAnalysisRequest(BaseModel):
    images: List[str]  # Lista de imágenes en base64
    ubicacion_exacta: Optional[str] = ""
    epoca_construccion: Optional[str] = ""
    uso_actual: Optional[str] = ""
    ultima_intervencion: Optional[str] = ""

class SingleImageAnalysisRequest(BaseModel):
    image: str  # Una sola imagen en base64
    ubicacion_exacta: Optional[str] = ""
    epoca_construccion: Optional[str] = ""
    uso_actual: Optional[str] = ""
    ultima_intervencion: Optional[str] = ""

class AnalysisResponse(BaseModel):
    success: bool
    message: str
    results: Optional[dict] = None
    reconstructed_facade: Optional[str] = None  # Imagen en base64
    damage_analysis: Optional[str] = None
    diagnosis: Optional[str] = None
class MultipleImagesAnalysisResponse(BaseModel):
    success: bool
    message: str
    results: dict
    images: dict  # Diccionario con las diferentes imágenes
    compression_info: dict  # Información sobre la compresión
    damage_analysis: Union[dict, str]
    diagnosis: str

@app.on_event("startup")
async def startup_event():
    """Inicializar modelos al arrancar la aplicación"""
    global models, transforms_list, diagnosis_system
    
    print("🚀 Inicializando modelos...")
    
    try:
        # Cargar modelos de análisis
        models = load_test_models()
        transforms_list = get_transforms()
        
        # Inicializar sistema de diagnóstico
        ruta_pdf = "D:/IA3/Datasets/Completo_160625_compressed.pdf"
        if os.path.exists(ruta_pdf):
            diagnosis_system = DiagnosisSystem(ruta_pdf)
            diagnosis_system.initialize_llm(model_name="gpt-3.5-turbo", temperature=0.3)
            print("✅ Sistema de diagnóstico inicializado")
        else:
            print("⚠️ PDF de patologías no encontrado, diagnóstico deshabilitado")
            
        print("✅ Modelos cargados exitosamente")
        
    except Exception as e:
        print(f"❌ Error al cargar modelos: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Facade Analysis API - Sistema de análisis de fachadas"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": models is not None,
        "diagnosis_available": diagnosis_system is not None
    }

@app.post("/analyze/multiple", response_model=MultipleImagesAnalysisResponse)
async def analyze_multiple_images_improved(request: ImageAnalysisRequest):
    """
    Versión mejorada que devuelve múltiples imágenes por separado
    """
    try:
        if len(request.images) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Se necesitan al menos 2 imágenes para la reconstrucción SFM"
            )
        
        print(f"📸 Procesando {len(request.images)} imágenes...")
        
        # Convertir imágenes base64 a OpenCV (TU CÓDIGO EXISTENTE)
        imagenes = []
        for i, img_base64 in enumerate(request.images):
            try:
                img_rgb = base64_to_cv2(img_base64)
                #img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                imagenes.append(img_rgb)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error al procesar imagen {i+1}: {str(e)}"
                )
        
        # PARTE 1: Reconstrucción SFM (TU CÓDIGO EXISTENTE)
        print("🔄 Iniciando reconstrucción SFM...")
        
        keypoints, descriptores = detectar_caracteristicas(imagenes)
        todas_correspondencias, buenos_matches = encontrar_correspondencias(descriptores)
        homografias, mascaras = estimar_homografias(keypoints, buenos_matches)
        h_compuestas = componer_homografias(homografias, imagenes)
        x_min, y_min, x_max, y_max = calcular_limites_panorama(imagenes, h_compuestas)
        panorama = crear_mosaico(imagenes, h_compuestas, x_min, y_min, x_max, y_max)
        
        print("✅ Reconstrucción SFM completada")
        
        # PARTE 2: Análisis mejorado con modelos de IA
        print("🧠 Analizando con modelos de IA mejorados...")
        
        #plano, info = prepare_image_for_analysis(panorama, DAMAGE_DETECTION_SIZE, 'adaptive')
        plano = panorama
        facade_img, facade_visualization, damage_mask, element_mask, damage_patches, damage_locations = \
            improved_load_and_analyze_facade(plano, models, transforms_list)
        
        # Generar análisis mejorado
        resultados_analisis = generate_damage_analysis_improved(damage_mask, element_mask)
        
        print("✅ Análisis con IA completado")
        
        # PARTE 3: Crear imágenes individuales
        print("🖼️ Generando imágenes individuales...")
        
        analysis_images = create_individual_analysis_images(
            facade_img, facade_visualization, damage_mask, element_mask
        )
        
        # PARTE 4: Compresión y codificación de múltiples imágenes
        print("🗜️ Comprimiendo múltiples imágenes para transmisión...")
        
        encoded_images, compression_info = compress_and_encode_images(
            analysis_images, 
            {'panorama_shape': panorama.shape}
        )
        
        # PARTE 5: Diagnóstico
        diagnostico = "No se generó diagnóstico"  # Valor por defecto
        if diagnosis_system is not None:
            print("🩺 Generando diagnóstico...")
            try:
                # Usar el análisis mejorado para el diagnóstico
                diagnostico = diagnosis_system.generar_diagnostico(
                    resultados_analisis=resultados_analisis,
                    ubicacion_exacta=request.ubicacion_exacta,
                    epoca_construccion=request.epoca_construccion,
                    uso_actual=request.uso_actual,
                    ultima_intervencion=request.ultima_intervencion
                )
                print("✅ Diagnóstico generado")
            except Exception as e:
                print(f"⚠ Error en diagnóstico: {str(e)}")
                diagnostico = f"Error al generar diagnóstico: {str(e)}"
        
        # ========== NUEVA SECCIÓN: COMPRESIÓN INTELIGENTE DE IMAGEN ==========
        print("🗜️ Comprimiendo imagen para transmisión...")
        
        # Función para comprimir imagen inteligentemente
        return MultipleImagesAnalysisResponse(
            success=True,
            message="Análisis completado exitosamente con múltiples imágenes",
            results={
                "num_images_processed": len(request.images),
                "panorama_dimensions": f"{panorama.shape[1]}x{panorama.shape[0]}",
                "num_output_images": len(encoded_images),
                "available_images": list(encoded_images.keys()),
                "damage_analysis": resultados_analisis,
            },
            images=encoded_images,
            compression_info=compression_info,
            damage_analysis=resultados_analisis,
            diagnosis=diagnostico
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en análisis múltiple: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/analyze/single", response_model=MultipleImagesAnalysisResponse)
async def analyze_single_image(request: SingleImageAnalysisRequest):
    """
    Analiza una sola imagen y devuelve múltiples visualizaciones
    """
    try:
        print("📸 Procesando imagen única...")
        # Convertir imagen base64 a OpenCV
        #img_cv2 = base64_to_cv2(request.image)
        #img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_rgb = base64_to_cv2(request.image)
        #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print("🧠 Analizando con modelos de IA...")
        
        # Preparar imagen para análisis
        # plano = img_rgb
        plano, info = prepare_image_for_analysis(img_rgb, DAMAGE_DETECTION_SIZE, 'adaptive')
        config = get_adaptive_config_for_panorama(plano.shape)
        
        # Análisis con modelos
        facade_img, facade_visualization, damage_mask, element_mask, damage_patches, damage_locations = \
            improved_load_and_analyze_facade(plano, models, transforms_list)
        
        # Generar análisis mejorado
        resultados_analisis = generate_damage_analysis_improved(damage_mask, element_mask)
        
        print("✅ Análisis con IA completado")
        
        # Crear imágenes individuales
        print("🖼️ Generando imágenes individuales...")
        
        analysis_images = create_individual_analysis_images(
            facade_img, facade_visualization, damage_mask, element_mask
        )
        
        # Compresión y codificación de múltiples imágenes
        print("🗜️ Comprimiendo múltiples imágenes para transmisión...")
        
        encoded_images, compression_info = compress_and_encode_images(
            analysis_images, 
            {'original_shape': img_rgb.shape}
        )
        
        # Diagnóstico
        diagnostico = "No se generó diagnóstico"  # Valor por defecto
        
        if diagnosis_system is not None:
            print("🩺 Generando diagnóstico...")
            try:
                diagnostico = diagnosis_system.generar_diagnostico(
                    resultados_analisis=resultados_analisis,
                    ubicacion_exacta=request.ubicacion_exacta,
                    epoca_construccion=request.epoca_construccion,
                    uso_actual=request.uso_actual,
                    ultima_intervencion=request.ultima_intervencion
                )
                print("✅ Diagnóstico generado")
            except Exception as e:
                print(f"⚠️ Error en diagnóstico: {str(e)}")
                diagnostico = f"Error al generar diagnóstico: {str(e)}"
        # ========== NUEVA SECCIÓN: COMPRESIÓN INTELIGENTE DE IMAGEN ==========
        print("🗜️ Comprimiendo imagen para transmisión...")
        
        return MultipleImagesAnalysisResponse(
            success=True,
            message="Análisis de imagen única completado con múltiples visualizaciones",
            results={
                "original_dimensions": f"{img_rgb.shape[1]}x{img_rgb.shape[0]}",
                "processed_dimensions": f"{facade_visualization.shape[1]}x{facade_visualization.shape[0]}",
                "num_output_images": len(encoded_images),
                "available_images": list(encoded_images.keys()),
                "damage_analysis": resultados_analisis
            },
            images=encoded_images,
            compression_info=compression_info,
            damage_analysis=resultados_analisis,
            diagnosis=diagnostico
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en análisis único: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/test/echo")
async def test_echo(request: dict):
    """Endpoint de prueba para verificar conectividad"""
    return {
        "success": True,
        "message": "Echo test successful",
        "received_data": request
    }

if __name__ == "_main_":
    import uvicorn
    try:
        print("Iniciando el servidor...")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="debug")
    except Exception as e:
        print(f"Error al iniciar el servidor: {str(e)}")
        import traceback
        traceback.print_exc()
        #uvicorn app:app