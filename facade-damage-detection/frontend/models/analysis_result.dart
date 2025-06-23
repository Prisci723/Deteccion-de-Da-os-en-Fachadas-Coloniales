class AnalysisResult {
  final String message;
  final Map<String, dynamic>? images;
  final dynamic damageAnalysis;
  final String? diagnosis;
  final Map<String, dynamic>? compressionInfo;
  final Map<String, dynamic>? results;
  final String? reconstructedFacade;

  AnalysisResult({
    required this.message,
    this.images,
    this.damageAnalysis,
    this.diagnosis,
    this.compressionInfo,
    this.results,
    this.reconstructedFacade,
  });

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      message: json['message'] ?? '',
      images: json['images'] as Map<String, dynamic>?,
      damageAnalysis: json['damage_analysis'],
      diagnosis: json['diagnosis']?.toString(),
      compressionInfo: json['compression_info'] as Map<String, dynamic>?,
      results: json['results'] as Map<String, dynamic>?,
      reconstructedFacade: json['reconstructed_facade'] as String?,
    );
  }
}
