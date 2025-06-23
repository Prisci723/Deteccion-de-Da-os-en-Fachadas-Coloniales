import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import '../models/analysis_result.dart';

class ImageAnalysisService {
  final ImagePicker _picker = ImagePicker();
  static const String baseUrl = 'https://v5t24nrh-8000.brs.devtunnels.ms';

  Future<List<XFile>> pickImages() async {
    try {
      final List<XFile> images = await _picker.pickMultiImage();
      return images;
    } catch (e) {
      throw Exception('Error al seleccionar imágenes: $e');
    }
  }

  Future<XFile?> takePicture() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.camera);
      return image;
    } catch (e) {
      throw Exception('Error al tomar foto: $e');
    }
  }

  Future<String> imageToBase64(XFile imageFile) async {
    final bytes = await imageFile.readAsBytes();
    return base64Encode(bytes);
  }

  Future<AnalysisResult?> analyzeImages({
    required List<XFile> images,
    String? ubicacion,
    String? epoca,
    String? uso,
    String? intervencion,
  }) async {
    if (images.isEmpty) {
      throw Exception('Por favor selecciona al menos una imagen');
    }

    final List<String> base64Images = [];
    for (XFile image in images) {
      final base64Image = await imageToBase64(image);
      base64Images.add(base64Image);
    }

    try {
      String endpoint;
      Map<String, dynamic> requestBody;

      if (base64Images.length == 1) {
        endpoint = '$baseUrl/analyze/single';
        requestBody = {
          'image': base64Images.first,
          'ubicacion_exacta': ubicacion?.isNotEmpty == true ? ubicacion : null,
          'epoca_construccion': epoca?.isNotEmpty == true ? epoca : null,
          'uso_actual': uso?.isNotEmpty == true ? uso : null,
          'ultima_intervencion':
              intervencion?.isNotEmpty == true ? intervencion : null,
        };
      } else {
        endpoint = '$baseUrl/analyze/multiple';
        requestBody = {
          'images': base64Images,
          'ubicacion_exacta': ubicacion?.isNotEmpty == true ? ubicacion : null,
          'epoca_construccion': epoca?.isNotEmpty == true ? epoca : null,
          'uso_actual': uso?.isNotEmpty == true ? uso : null,
          'ultima_intervencion':
              intervencion?.isNotEmpty == true ? intervencion : null,
        };
      }

      final response = await http.post(
        Uri.parse(endpoint),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(requestBody),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);
        if (responseData['success'] == true) {
          return AnalysisResult.fromJson(responseData);
        } else {
          throw Exception('Error en el análisis: ${responseData['message']}');
        }
      } else {
        throw Exception('Error del servidor: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error de conexión: $e');
    }
  }
}
