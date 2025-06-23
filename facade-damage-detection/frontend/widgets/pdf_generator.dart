import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import '../models/analysis_result.dart';
import '../utils/helpers.dart';

class PDFGenerator extends StatelessWidget {
  final AnalysisResult analysisResult;
  final String? ubicacion;
  final String? epoca;
  final String? uso;
  final String? intervencion;

  const PDFGenerator({
    super.key,
    required this.analysisResult,
    this.ubicacion,
    this.epoca,
    this.uso,
    this.intervencion,
  });

  Future<void> _generatePDF(BuildContext context) async {
    try {
      final pdf = pw.Document();
      final now = DateTime.now();
      final formattedDate = '${now.day}/${now.month}/${now.year}';

      final List<pw.Widget> allContent = await _buildSafeContent(formattedDate);

      pdf.addPage(
        pw.MultiPage(
          pageFormat: PdfPageFormat.a4,
          margin: const pw.EdgeInsets.all(32),
          maxPages: 20,
          header:
              (context) => pw.Container(
                alignment: pw.Alignment.centerRight,
                child: pw.Text(
                  'Análisis de Fachada - $formattedDate',
                  style: pw.TextStyle(fontSize: 10, color: PdfColors.grey),
                ),
              ),
          footer:
              (context) => pw.Container(
                alignment: pw.Alignment.centerRight,
                child: pw.Text(
                  'Página ${context.pageNumber} de ${context.pagesCount}',
                  style: pw.TextStyle(fontSize: 10, color: PdfColors.grey),
                ),
              ),
          build: (pw.Context context) => allContent,
        ),
      );

      if (analysisResult.images != null) {
        _addImagePagesWithLimit(pdf, formattedDate);
      }

      await Printing.layoutPdf(
        onLayout: (PdfPageFormat format) async => pdf.save(),
        name: 'Analisis_Fachada_$formattedDate.pdf',
      );
    } catch (e) {
      _showError(context, 'Error al generar PDF: $e');
    }
  }

  Future<List<pw.Widget>> _buildSafeContent(String formattedDate) async {
    final List<pw.Widget> content = [];

    try {
      content.add(_buildPDFHeader(formattedDate));
      content.addAll(_buildPropertyInfoPDF());
      content.addAll(_buildDiagnosisSectionPDFSafe());
      content.addAll(_buildDamageAnalysisSectionPDFSafe());
      // content.add(_buildSummarySection());
    } catch (e) {
      content.add(
        pw.Container(
          padding: const pw.EdgeInsets.all(16),
          decoration: pw.BoxDecoration(
            color: PdfColors.red50,
            border: pw.Border.all(color: PdfColors.red),
          ),
          child: pw.Text(
            'Error al procesar contenido: $e',
            style: pw.TextStyle(color: PdfColors.red),
          ),
        ),
      );
    }

    return content;
  }

  pw.Widget _buildPDFHeader(String formattedDate) {
    return pw.Container(
      width: double.infinity,
      padding: const pw.EdgeInsets.all(16),
      margin: const pw.EdgeInsets.only(bottom: 20),
      decoration: pw.BoxDecoration(
        color: PdfColors.blue50,
        borderRadius: pw.BorderRadius.circular(8),
        border: pw.Border.all(color: PdfColors.blue200),
      ),
      child: pw.Column(
        crossAxisAlignment: pw.CrossAxisAlignment.start,
        mainAxisSize: pw.MainAxisSize.min,
        children: [
          pw.Text(
            'Análisis de la Fachada',
            style: pw.TextStyle(
              fontSize: 20,
              fontWeight: pw.FontWeight.bold,
              color: PdfColors.blue800,
            ),
          ),
          pw.SizedBox(height: 4),
          pw.Text(
            formattedDate,
            style: pw.TextStyle(fontSize: 14, color: PdfColors.grey700),
          ),
        ],
      ),
    );
  }

  List<pw.Widget> _buildPropertyInfoPDF() {
    if (ubicacion?.isEmpty == true &&
        epoca?.isEmpty == true &&
        uso?.isEmpty == true &&
        intervencion?.isEmpty == true) {
      return [];
    }

    List<pw.Widget> infoFields = [];

    if (ubicacion?.isNotEmpty == true) {
      infoFields.add(_buildInfoRow('Ubicación:', ubicacion!));
    }
    if (epoca?.isNotEmpty == true) {
      infoFields.add(_buildInfoRow('Época de Construcción:', epoca!));
    }
    if (uso?.isNotEmpty == true) {
      infoFields.add(_buildInfoRow('Uso Actual:', uso!));
    }
    if (intervencion?.isNotEmpty == true) {
      infoFields.add(_buildInfoRow('Última Intervención:', intervencion!));
    }

    return [
      pw.Container(
        margin: const pw.EdgeInsets.only(bottom: 10),
        child: pw.Text(
          'Información del Inmueble',
          style: pw.TextStyle(
            fontSize: 16,
            fontWeight: pw.FontWeight.bold,
            color: PdfColors.blue800,
          ),
        ),
      ),
      pw.Container(
        width: double.infinity,
        padding: const pw.EdgeInsets.all(12),
        margin: const pw.EdgeInsets.only(bottom: 15),
        decoration: pw.BoxDecoration(
          color: PdfColors.grey50,
          borderRadius: pw.BorderRadius.circular(6),
          border: pw.Border.all(color: PdfColors.grey300),
        ),
        child: pw.Column(
          crossAxisAlignment: pw.CrossAxisAlignment.start,
          mainAxisSize: pw.MainAxisSize.min,
          children: _buildInfoFieldsWithSpacing(infoFields),
        ),
      ),
    ];
  }

  void _addImagePagesWithLimit(pw.Document pdf, String formattedDate) {
    try {
      final images = analysisResult.images!;
      int imageCount = 0;
      const int maxImages = 5;

      for (var entry in images.entries) {
        if (imageCount >= maxImages) {
          pdf.addPage(
            pw.Page(
              pageFormat: PdfPageFormat.a4,
              margin: const pw.EdgeInsets.all(32),
              build: (pw.Context context) {
                return pw.Center(
                  child: pw.Container(
                    padding: const pw.EdgeInsets.all(20),
                    decoration: pw.BoxDecoration(
                      color: PdfColors.orange50,
                      border: pw.Border.all(color: PdfColors.orange),
                      borderRadius: pw.BorderRadius.circular(8),
                    ),
                    child: pw.Column(
                      mainAxisSize: pw.MainAxisSize.min,
                      children: [
                        pw.Text(
                          'Límite de Imágenes Alcanzado',
                          style: pw.TextStyle(
                            fontSize: 16,
                            fontWeight: pw.FontWeight.bold,
                            color: PdfColors.orange800,
                          ),
                        ),
                        pw.SizedBox(height: 10),
                        pw.Text(
                          'Se han incluido las primeras $maxImages imágenes del análisis.\nPara ver todas las imágenes, consulte la aplicación.',
                          style: const pw.TextStyle(fontSize: 12),
                          textAlign: pw.TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          );
          break;
        }

        try {
          final imageBytes = base64Decode(entry.value);

          pdf.addPage(
            pw.Page(
              pageFormat: PdfPageFormat.a4,
              margin: const pw.EdgeInsets.all(32),
              build: (pw.Context context) {
                return pw.Column(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: [
                    pw.Container(
                      width: double.infinity,
                      padding: const pw.EdgeInsets.all(15),
                      decoration: pw.BoxDecoration(
                        color: PdfColors.blue50,
                        borderRadius: pw.BorderRadius.circular(8),
                        border: pw.Border.all(color: PdfColors.blue200),
                      ),
                      child: pw.Column(
                        crossAxisAlignment: pw.CrossAxisAlignment.start,
                        children: [
                          pw.Text(
                            'Imagen de Análisis ${imageCount + 1}',
                            style: pw.TextStyle(
                              fontSize: 18,
                              fontWeight: pw.FontWeight.bold,
                              color: PdfColors.blue800,
                            ),
                          ),
                          pw.SizedBox(height: 5),
                          pw.Text(
                            getImageTitle(entry.key),
                            style: pw.TextStyle(
                              fontSize: 16,
                              fontWeight: pw.FontWeight.bold,
                            ),
                          ),
                          pw.SizedBox(height: 3),
                          pw.Text(
                            getImageDescription(entry.key),
                            style: pw.TextStyle(
                              fontSize: 11,
                              color: PdfColors.grey700,
                            ),
                          ),
                        ],
                      ),
                    ),
                    pw.SizedBox(height: 20),
                    pw.Expanded(
                      child: pw.Container(
                        width: double.infinity,
                        alignment: pw.Alignment.center,
                        child: pw.Container(
                          decoration: pw.BoxDecoration(
                            border: pw.Border.all(color: PdfColors.grey300),
                            borderRadius: pw.BorderRadius.circular(8),
                          ),
                          child: pw.ClipRRect(
                            horizontalRadius: 8,
                            verticalRadius: 8,
                            child: pw.Image(
                              pw.MemoryImage(imageBytes),
                              fit: pw.BoxFit.contain,
                            ),
                          ),
                        ),
                      ),
                    ),
                    pw.SizedBox(height: 20),
                    pw.Divider(),
                    pw.Text(
                      'Análisis de Fachada - $formattedDate',
                      style: pw.TextStyle(
                        fontSize: 10,
                        color: PdfColors.grey600,
                      ),
                    ),
                  ],
                );
              },
            ),
          );

          imageCount++;
        } catch (e) {
          continue;
        }
      }
    } catch (e) {
      print('Error al procesar imágenes para PDF: $e');
    }
  }

  List<pw.Widget> _buildInfoFieldsWithSpacing(List<pw.Widget> infoFields) {
    List<pw.Widget> fieldsWithSpacing = [];

    for (int i = 0; i < infoFields.length; i++) {
      fieldsWithSpacing.add(infoFields[i]);
      if (i < infoFields.length - 1) {
        fieldsWithSpacing.add(pw.SizedBox(height: 6));
      }
    }

    return fieldsWithSpacing;
  }

  pw.Widget _buildInfoRow(String label, String value) {
    return pw.Row(
      crossAxisAlignment: pw.CrossAxisAlignment.start,
      children: [
        pw.Container(
          width: 120,
          child: pw.Text(
            label,
            style: pw.TextStyle(fontWeight: pw.FontWeight.bold, fontSize: 12),
          ),
        ),
        pw.Expanded(
          child: pw.Text(
            value.length > 100 ? value.substring(0, 100) + '...' : value,
            style: const pw.TextStyle(fontSize: 12),
            softWrap: true,
          ),
        ),
      ],
    );
  }

  List<pw.Widget> _buildDiagnosisSectionPDFSafe() {
    if (analysisResult.diagnosis == null ||
        analysisResult.diagnosis == 'No se generó diagnóstico') {
      return [];
    }

    String diagnosisText = analysisResult.diagnosis!;

    if (diagnosisText.length > 5000) {
      diagnosisText =
          diagnosisText.substring(0, 4950) + '... [Contenido truncado]';
    }

    return [
      pw.Container(
        margin: const pw.EdgeInsets.only(bottom: 10),
        child: pw.Text(
          'Diagnóstico',
          style: pw.TextStyle(
            fontSize: 16,
            fontWeight: pw.FontWeight.bold,
            color: PdfColors.blue800,
          ),
        ),
      ),
      ..._buildTextInChunksSafe(
        diagnosisText,
        PdfColors.blue50,
        PdfColors.blue200,
      ),
      pw.SizedBox(height: 15),
    ];
  }

  List<pw.Widget> _buildDamageAnalysisSectionPDFSafe() {
    if (analysisResult.damageAnalysis == null) {
      return [];
    }

    String damageText = _getDamageAnalysisTextSafe();

    return [
      pw.Container(
        margin: const pw.EdgeInsets.only(bottom: 10),
        child: pw.Text(
          'Análisis de Daños',
          style: pw.TextStyle(
            fontSize: 16,
            fontWeight: pw.FontWeight.bold,
            color: PdfColors.blue800,
          ),
        ),
      ),
      ..._buildTextInChunksSafe(
        damageText,
        PdfColors.orange50,
        PdfColors.orange200,
      ),
      pw.SizedBox(height: 15),
    ];
  }

  String _getDamageAnalysisTextSafe() {
    final damageAnalysis = analysisResult.damageAnalysis;
    String text = '';

    if (damageAnalysis is String) {
      text = damageAnalysis;
    } else if (damageAnalysis is Map) {
      if (damageAnalysis['summary'] != null) {
        text = damageAnalysis['summary'].toString();
      } else {
        text = const JsonEncoder.withIndent('  ').convert(damageAnalysis);
      }
    } else {
      text = 'No se encontró análisis de daños';
    }

    if (text.length > 4000) {
      text = text.substring(0, 3950) + '... [Contenido truncado]';
    }

    return text;
  }

  List<pw.Widget> _buildTextInChunksSafe(
    String text,
    PdfColor backgroundColor,
    PdfColor borderColor,
  ) {
    const int maxChunks = 3;
    const int maxCharsPerChunk = 1500;

    final List<pw.Widget> widgets = [];

    if (text.length <= maxCharsPerChunk) {
      widgets.add(
        _createSafeTextChunk(text, backgroundColor, borderColor, 1, false),
      );
    } else {
      List<String> chunks = _splitTextIntoSafeChunks(
        text,
        maxCharsPerChunk,
        maxChunks,
      );

      for (int i = 0; i < chunks.length; i++) {
        widgets.add(
          _createSafeTextChunk(
            chunks[i],
            backgroundColor,
            borderColor,
            i + 1,
            chunks.length > 1,
          ),
        );
      }
    }

    return widgets;
  }

  List<String> _splitTextIntoSafeChunks(
    String text,
    int maxCharsPerChunk,
    int maxChunks,
  ) {
    List<String> chunks = [];

    String workingText = text;
    int maxTotalChars = maxCharsPerChunk * maxChunks;

    if (workingText.length > maxTotalChars) {
      workingText =
          workingText.substring(0, maxTotalChars - 50) +
          '... [Texto truncado por seguridad]';
    }

    int start = 0;
    while (start < workingText.length && chunks.length < maxChunks) {
      int end = start + maxCharsPerChunk;

      if (end >= workingText.length) {
        chunks.add(workingText.substring(start));
        break;
      }

      int cutPoint = end;
      for (int i = end - 100; i > start + 100 && i < end; i++) {
        if (workingText[i] == '.' || workingText[i] == '\n') {
          cutPoint = i + 1;
          break;
        }
      }

      chunks.add(workingText.substring(start, cutPoint));
      start = cutPoint;
    }

    return chunks;
  }

  pw.Widget _createSafeTextChunk(
    String text,
    PdfColor backgroundColor,
    PdfColor borderColor,
    int chunkIndex,
    bool isMultiChunk,
  ) {
    return pw.Container(
      width: double.infinity,
      padding: const pw.EdgeInsets.all(12),
      margin: const pw.EdgeInsets.only(bottom: 8),
      decoration: pw.BoxDecoration(
        color: backgroundColor,
        border: pw.Border.all(color: borderColor),
        borderRadius: pw.BorderRadius.circular(6),
      ),
      child: pw.Column(
        crossAxisAlignment: pw.CrossAxisAlignment.start,
        mainAxisSize: pw.MainAxisSize.min,
        children: [
          if (isMultiChunk && chunkIndex > 1) ...[
            pw.Text(
              '--- Parte $chunkIndex ---',
              style: pw.TextStyle(
                fontSize: 9,
                color: PdfColors.grey600,
                fontStyle: pw.FontStyle.italic,
              ),
            ),
            pw.SizedBox(height: 5),
          ],
          pw.Text(
            text,
            style: const pw.TextStyle(fontSize: 11),
            textAlign: pw.TextAlign.justify,
          ),
        ],
      ),
    );
  }

  // pw.Widget _buildSummarySection() {
  //   return pw.Container(
  //     width: double.infinity,
  //     padding: const pw.EdgeInsets.all(12),
  //     margin: const pw.EdgeInsets.only(top: 10),
  //     decoration: pw.BoxDecoration(
  //       color: PdfColors.green50,
  //       border: pw.Border.all(color: PdfColors.green200),
  //       borderRadius: pw.BorderRadius.circular(6),
  //     ),
  //     // child: pw.Column(
  //     //   crossAxisAlignment: pw.CrossAxisAlignment.start,
  //     //   mainAxisSize: pw.MainAxisSize.min,
  //     //   children: [
  //     //     // pw.Text(
  //     //     //   'Resumen del Análisis',
  //     //     //   style: pw.TextStyle(
  //     //     //     fontSize: 14,
  //     //     //     fontWeight: pw.FontWeight.bold,
  //     //     //     color: PdfColors.green800,
  //     //     //   ),
  //     //     // ),
  //     //     pw.SizedBox(height: 8),
  //     //     pw.Text(
  //     //       'Este reporte contiene el análisis detallado de la fachada con las imágenes procesadas y recomendaciones técnicas correspondientes.',
  //     //       style: const pw.TextStyle(fontSize: 10),
  //     //     ),
  //     //     pw.SizedBox(height: 6),
  //     //     pw.Text(
  //     //       'Estado: ${analysisResult.message}',
  //     //       style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold),
  //     //     ),
  //     //   ],
  //     // ),
  //   );
  // }

  void _showError(BuildContext context, String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(top: 16),
      child: ElevatedButton.icon(
        onPressed: () => _generatePDF(context),
        icon: const Icon(Icons.picture_as_pdf),
        label: const Text('Generar PDF'),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.green,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 12),
        ),
      ),
    );
  }
}
