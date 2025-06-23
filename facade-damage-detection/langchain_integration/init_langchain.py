import os
import pickle
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

class DiagnosisSystem:
    """Sistema de diagnóstico optimizado con cache de embeddings"""
    
    def __init__(self, pdf_path, cache_dir="./cache"):
        self.pdf_path = pdf_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Rutas de cache
        self.vectorstore_cache_path = self.cache_dir / "vectorstore.faiss"
        self.embeddings_cache_path = self.cache_dir / "embeddings.pkl"
        self.documents_cache_path = self.cache_dir / "documents.pkl"
        
        # Componentes del sistema
        self.vectorstore = None
        self.embeddings = None
        self.llm = None
        
        # Inicializar sistema
        self._initialize_system()
    
    def _get_pdf_hash(self):
        """Obtiene un hash del PDF para detectar cambios"""
        import hashlib
        with open(self.pdf_path, 'rb') as f:
            pdf_hash = hashlib.md5(f.read()).hexdigest()
        return pdf_hash
    
    def _is_cache_valid(self):
        """Verifica si el cache es válido comparando el hash del PDF"""
        hash_file = self.cache_dir / "pdf_hash.txt"
        
        if not hash_file.exists():
            return False
            
        try:
            with open(hash_file, 'r') as f:
                cached_hash = f.read().strip()
            current_hash = self._get_pdf_hash()
            return cached_hash == current_hash
        except:
            return False
    
    def _save_pdf_hash(self):
        """Guarda el hash del PDF actual"""
        hash_file = self.cache_dir / "pdf_hash.txt"
        current_hash = self._get_pdf_hash()
        with open(hash_file, 'w') as f:
            f.write(current_hash)
    
    def _load_from_cache(self):
        """Carga los componentes desde el cache"""
        try:
            # Cargar embeddings
            with open(self.embeddings_cache_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            
            # Cargar vectorstore
            self.vectorstore = FAISS.load_local(
                str(self.vectorstore_cache_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            print("✅ Embeddings y vectorstore cargados desde cache")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando desde cache: {e}")
            return False
    
    def _save_to_cache(self):
        """Guarda los componentes en cache"""
        try:
            # Guardar embeddings
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Guardar vectorstore
            self.vectorstore.save_local(str(self.vectorstore_cache_path))
            
            # Guardar hash del PDF
            self._save_pdf_hash()
            
            print("✅ Embeddings y vectorstore guardados en cache")
            
        except Exception as e:
            print(f"❌ Error guardando en cache: {e}")
    
    def _create_fresh_vectorstore(self):
        """Crea un nuevo vectorstore desde el PDF"""
        print("🔄 Creando nueva base de conocimiento...")
        
        # Cargar PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        # Dividir en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Crear embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # Crear vectorstore
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        print(f"✅ Base de conocimiento creada con {len(chunks)} fragmentos")
        
        # Guardar en cache
        self._save_to_cache()
    
    def _initialize_system(self):
        """Inicializa el sistema cargando desde cache o creando nuevo"""
        # Verificar si existe cache válido
        if (self._is_cache_valid() and 
            self.vectorstore_cache_path.exists() and 
            self.embeddings_cache_path.exists()):
            
            print("🔄 Intentando cargar desde cache...")
            if self._load_from_cache():
                return
        
        # Si no hay cache válido, crear nuevo
        print("🔄 Cache no válido o inexistente, creando nuevo...")
        self._create_fresh_vectorstore()
    
    def initialize_llm(self, model_name="gpt-4", temperature=0.3):
        """Inicializa el modelo LLM"""
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        print(f"✅ LLM inicializado: {model_name}")
    

    def obtener_contexto_relevante(self, resultados_analisis, elementos_arquitectonicos):
        """Obtiene información contextual relevante de la base de conocimiento"""
        if not self.vectorstore:
            raise ValueError("Vectorstore no inicializado")
       
        # Construir consulta
        consulta = f"Información sobre daños en {', '.join(elementos_arquitectonicos)} "
        consulta += f"incluyendo {', '.join([d for d in ['desprendimiento', 'humedad', 'deterioro', 'hundimiento'] if d in resultados_analisis.lower()])}"
       
        # Recuperar documentos relevantes
        documentos_relevantes = self.vectorstore.similarity_search(consulta, k=3)
        contexto = "\n\n".join([doc.page_content for doc in documentos_relevantes])
       
        return contexto
   
    def crear_plantilla_prompt(self):
        """Crea la plantilla de prompt para diagnósticos"""
        from langchain.prompts import PromptTemplate
       
        template = """
        Eres un experto en diagnóstico de patologías en fachadas. Basándote en los resultados del análisis
        y la información de referencia, recibes danos en base a la cantidad de pixeles de una imagen por lo que debes calcular la severidad, genera un diagnóstico técnico completo.
        RESULTADOS DEL ANÁLISIS:
        {resultados_analisis}
        INFORMACIÓN CONTEXTUAL:
        Ubicación: {ubicacion_exacta}
        Época de construcción: {epoca_construccion}
        Uso actual: {uso_actual}
        Última intervención: {ultima_intervencion}
        INFORMACIÓN DE REFERENCIA:
        {contexto_adicional}
        Genera un diagnóstico que incluya:
        1. Diagnostico general del estado de la fachada
        2. Causas probables del estado general
        3. Identificación de patologías por elemento detectadas
        4. Recomendaciones de intervención
        5. Medidas preventivas
        DIAGNÓSTICO:
        """
       
        return PromptTemplate(
            input_variables=[
                "resultados_analisis", "ubicacion_exacta", "epoca_construccion",
                "uso_actual", "ultima_intervencion", "contexto_adicional"
            ],
            template=template
        )
   
    def generar_diagnostico(self, resultados_analisis, ubicacion_exacta, epoca_construccion,
                           uso_actual, ultima_intervencion, temperatura=0.3):
        """Genera un diagnóstico completo basado en los resultados del análisis"""
       
        if not self.llm:
            raise ValueError("LLM no inicializado. Llama a initialize_llm() primero.")
       
        # Extraer elementos arquitectónicos
        elementos_arquitectonicos = []
        for linea in resultados_analisis.split('\n'):
            if 'ELEMENTO:' in linea:
                elementos_arquitectonicos.append(linea.split('ELEMENTO:')[1].strip().split(' ')[0])
       
        # Obtener contexto relevante
        contexto_adicional = self.obtener_contexto_relevante(resultados_analisis, elementos_arquitectonicos)
       
        # Configurar temperatura del LLM
        if hasattr(self.llm, 'temperature'):
            self.llm.temperature = temperatura
       
        # Crear cadena de diagnóstico
        prompt_template = self.crear_plantilla_prompt()
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
       
        # Generar diagnóstico
        respuesta = chain.run({
            "resultados_analisis": resultados_analisis,
            "ubicacion_exacta": ubicacion_exacta,
            "epoca_construccion": epoca_construccion,
            "uso_actual": uso_actual,
            "ultima_intervencion": ultima_intervencion,
            "contexto_adicional": contexto_adicional
        })
       
        return respuesta
    def invalidar_cache(self):
        """Invalida y elimina el cache existente"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            print("🗑️ Cache invalidado y eliminado")
    
    def get_stats(self):
        """Obtiene estadísticas del sistema"""
        stats = {
            "cache_exists": self.vectorstore_cache_path.exists(),
            "cache_valid": self._is_cache_valid(),
            "vectorstore_loaded": self.vectorstore is not None,
            "embeddings_loaded": self.embeddings is not None,
            "llm_loaded": self.llm is not None
        }
        return stats
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
