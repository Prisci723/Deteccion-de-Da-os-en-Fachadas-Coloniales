import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/image_analysis_service.dart';
import '../widgets/image_preview.dart';
import '../widgets/processed_image.dart';
import '../models/analysis_result.dart';
import 'analysis_results_screen.dart';

class ImageAnalysisScreen extends StatefulWidget {
  const ImageAnalysisScreen({super.key});

  @override
  State<ImageAnalysisScreen> createState() => _ImageAnalysisScreenState();
}

class _ImageAnalysisScreenState extends State<ImageAnalysisScreen> {
  final ImageAnalysisService _service = ImageAnalysisService();
  List<XFile> _selectedImages = [];
  bool _isAnalyzing = false;
  final TextEditingController _ubicacionController = TextEditingController();
  final TextEditingController _epocaController = TextEditingController();
  final TextEditingController _usoController = TextEditingController();
  final TextEditingController _intervencionController = TextEditingController();

  @override
  void dispose() {
    _ubicacionController.dispose();
    _epocaController.dispose();
    _usoController.dispose();
    _intervencionController.dispose();
    super.dispose();
  }

  Future<void> _pickImages() async {
    try {
      final images = await _service.pickImages();
      setState(() {
        _selectedImages = images;
      });
    } catch (e) {
      _showError(e.toString());
    }
  }

  Future<void> _takePicture() async {
    try {
      final image = await _service.takePicture();
      if (image != null) {
        setState(() {
          _selectedImages = [image];
        });
      }
    } catch (e) {
      _showError(e.toString());
    }
  }

  Future<void> _analyzeImages() async {
    setState(() {
      _isAnalyzing = true;
    });

    // Show loading dialog
    showDialog(
      context: context,
      barrierDismissible: false,
      builder:
          (context) => const Dialog(
            child: Padding(
              padding: EdgeInsets.all(20.0),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(width: 20),
                  Text('Analizando imágenes...'),
                ],
              ),
            ),
          ),
    );

    try {
      final result = await _service.analyzeImages(
        images: _selectedImages,
        ubicacion: _ubicacionController.text,
        epoca: _epocaController.text,
        uso: _usoController.text,
        intervencion: _intervencionController.text,
      );

      // Close loading dialog
      Navigator.of(context).pop();

      // Navigate to results screen
      Navigator.of(context).push(
        MaterialPageRoute(
          builder:
              (context) => AnalysisResultsScreen(
                analysisResult: result!,
                ubicacion: _ubicacionController.text,
                epoca: _epocaController.text,
                uso: _usoController.text,
                intervencion: _intervencionController.text,
              ),
        ),
      );
    } catch (e) {
      // Close loading dialog
      Navigator.of(context).pop();
      _showError(e.toString());
    } finally {
      setState(() {
        _isAnalyzing = false;
      });
    }
  }

  void _clearImages() {
    setState(() {
      _selectedImages.clear();
    });
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Análisis de Fachadas'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickImages,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Galería'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _takePicture,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Cámara'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _clearImages,
                    icon: const Icon(Icons.clear),
                    label: const Text('Limpiar'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            ImagePreview(images: _selectedImages),
            const SizedBox(height: 16),
            const Text(
              'Información Adicional (Opcional):',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _ubicacionController,
              decoration: const InputDecoration(
                labelText: 'Ubicación Exacta',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _epocaController,
              decoration: const InputDecoration(
                labelText: 'Época de Construcción',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _usoController,
              decoration: const InputDecoration(
                labelText: 'Uso Actual',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _intervencionController,
              decoration: const InputDecoration(
                labelText: 'Última Intervención',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _isAnalyzing ? null : _analyzeImages,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
              child:
                  _isAnalyzing
                      ? const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              valueColor: AlwaysStoppedAnimation<Color>(
                                Colors.white,
                              ),
                            ),
                          ),
                          SizedBox(width: 8),
                          Text('Analizando...'),
                        ],
                      )
                      : Text('Analizar ${_selectedImages.length} imagen(es)'),
            ),
          ],
        ),
      ),
    );
  }
}
