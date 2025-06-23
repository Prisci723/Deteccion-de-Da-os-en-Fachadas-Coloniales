import 'package:flutter/material.dart';
import '../models/analysis_result.dart';
import '../widgets/analysis_results.dart';
import '../widgets/processed_image.dart';

class AnalysisResultsScreen extends StatelessWidget {
  final AnalysisResult analysisResult;
  final String? ubicacion;
  final String? epoca;
  final String? uso;
  final String? intervencion;

  const AnalysisResultsScreen({
    super.key,
    required this.analysisResult,
    this.ubicacion,
    this.epoca,
    this.uso,
    this.intervencion,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Resultados del Análisis'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            AnalysisResults(
              analysisResult: analysisResult,
              ubicacion: ubicacion,
              epoca: epoca,
              uso: uso,
              intervencion: intervencion,
            ),
            const SizedBox(height: 16),
            ProcessedImage(base64Image: analysisResult.reconstructedFacade),
          ],
        ),
      ),
    );
  }
}
