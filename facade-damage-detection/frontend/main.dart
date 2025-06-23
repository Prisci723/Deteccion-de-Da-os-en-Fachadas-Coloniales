import 'package:flutter/material.dart';
import 'screens/image_analysis_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Análisis de Fachadas',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const ImageAnalysisScreen(),
    );
  }
}
