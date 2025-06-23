import 'dart:convert';
import 'package:flutter/material.dart';
import '../models/analysis_result.dart';
import '../utils/helpers.dart';
import 'pdf_generator.dart';

class AnalysisResults extends StatelessWidget {
  final AnalysisResult? analysisResult;
  final String? ubicacion;
  final String? epoca;
  final String? uso;
  final String? intervencion;

  const AnalysisResults({
    super.key,
    this.analysisResult,
    this.ubicacion,
    this.epoca,
    this.uso,
    this.intervencion,
  });

  @override
  Widget build(BuildContext context) {
    if (analysisResult == null) return const SizedBox.shrink();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Resultados del Análisis:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text('Estado: ${analysisResult!.message}'),
            const SizedBox(height: 16),
            if (analysisResult!.images != null) ...[
              _buildAnalysisImages(context),
              const SizedBox(height: 16),
            ],
            if (analysisResult!.damageAnalysis != null) ...[
              const Text(
                'Análisis de Daños:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              _buildDamageAnalysis(),
              const SizedBox(height: 16),
            ],
            if (analysisResult!.diagnosis != null &&
                analysisResult!.diagnosis != 'No se generó diagnóstico') ...[
              const Text(
                'Diagnóstico:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.blue.shade200),
                ),
                child: Text(
                  analysisResult!.diagnosis!,
                  style: const TextStyle(fontSize: 14),
                ),
              ),
              const SizedBox(height: 16),
            ],
            if (analysisResult!.compressionInfo != null) ...[
              _buildCompressionInfo(),
              const SizedBox(height: 16),
            ],
            if (analysisResult!.results != null) ...[_buildTechnicalDetails()],
            PDFGenerator(
              analysisResult: analysisResult!,
              ubicacion: ubicacion,
              epoca: epoca,
              uso: uso,
              intervencion: intervencion,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAnalysisImages(BuildContext context) {
    final images = analysisResult!.images!;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Imágenes de Análisis:',
          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        DefaultTabController(
          length: images.length,
          child: Column(
            children: [
              TabBar(
                tabs:
                    images.keys.map((key) {
                      String title = getImageTitle(key);
                      return Tab(
                        child: Text(
                          title,
                          style: const TextStyle(fontSize: 12),
                          textAlign: TextAlign.center,
                        ),
                      );
                    }).toList(),
                labelColor: Colors.blue,
                unselectedLabelColor: Colors.grey,
                indicatorColor: Colors.blue,
              ),
              SizedBox(
                height: 300,
                child: TabBarView(
                  children:
                      images.entries.map((entry) {
                        return _buildImageTab(context, entry.key, entry.value);
                      }).toList(),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildImageTab(BuildContext context, String key, String base64Image) {
    return Container(
      margin: const EdgeInsets.all(8),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(
              getImageDescription(key),
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: Colors.grey,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          Expanded(
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: ClipRRect(
                child: Image.memory(
                  base64Decode(base64Image),
                  fit: BoxFit.contain,
                  errorBuilder: (context, error, stackTrace) {
                    return Container(
                      alignment: Alignment.center,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.error_outline,
                            size: 48,
                            color: Colors.red.shade300,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Error al cargar imagen',
                            style: TextStyle(color: Colors.red.shade600),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          ElevatedButton.icon(
            onPressed: () => _showFullScreenImage(context, key, base64Image),
            icon: const Icon(Icons.fullscreen, size: 16),
            label: const Text('Ver completa'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blue.shade50,
              foregroundColor: Colors.blue.shade700,
              elevation: 0,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDamageAnalysis() {
    final damageAnalysis = analysisResult!.damageAnalysis;

    if (damageAnalysis is String) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.orange.shade50,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.orange.shade200),
        ),
        child: Text(damageAnalysis, style: const TextStyle(fontSize: 14)),
      );
    } else if (damageAnalysis is Map) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.orange.shade50,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.orange.shade200),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (damageAnalysis['summary'] != null) ...[
              Text(
                damageAnalysis['summary'].toString(),
                style: const TextStyle(fontSize: 14),
              ),
            ] else ...[
              Text(
                const JsonEncoder.withIndent('  ').convert(damageAnalysis),
                style: const TextStyle(fontSize: 12, fontFamily: 'monospace'),
              ),
            ],
          ],
        ),
      );
    }

    return const SizedBox.shrink();
  }

  Widget _buildCompressionInfo() {
    return ExpansionTile(
      title: const Text(
        'Información de Compresión',
        style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
      ),
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children:
                (analysisResult!.compressionInfo! as Map<String, dynamic>)
                    .entries
                    .map((entry) {
                      final info = entry.value as Map<String, dynamic>;
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 4),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              getImageTitle(entry.key),
                              style: const TextStyle(
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                            Text(
                              '${info['original_size_mb']}MB → ${info['final_size_mb']}MB (${info['compression_ratio']}x)',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.grey.shade600,
                              ),
                            ),
                          ],
                        ),
                      );
                    })
                    .toList(),
          ),
        ),
      ],
    );
  }

  Widget _buildTechnicalDetails() {
    return ExpansionTile(
      title: const Text(
        'Detalles Técnicos',
        style: TextStyle(fontWeight: FontWeight.bold),
      ),
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          child: Text(
            const JsonEncoder.withIndent('  ').convert(analysisResult!.results),
            style: const TextStyle(fontSize: 12, fontFamily: 'monospace'),
          ),
        ),
      ],
    );
  }

  void _showFullScreenImage(
    BuildContext context,
    String title,
    String base64Image,
  ) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder:
            (context) => Scaffold(
              appBar: AppBar(
                title: Text(getImageTitle(title)),
                backgroundColor: Colors.black,
                foregroundColor: Colors.white,
              ),
              backgroundColor: Colors.black,
              body: Center(
                child: InteractiveViewer(
                  panEnabled: true,
                  boundaryMargin: const EdgeInsets.all(20),
                  minScale: 0.5,
                  maxScale: 4.0,
                  child: Image.memory(
                    base64Decode(base64Image),
                    fit: BoxFit.contain,
                  ),
                ),
              ),
            ),
      ),
    );
  }
}
