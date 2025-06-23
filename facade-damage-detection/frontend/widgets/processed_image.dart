import 'dart:convert';
import 'package:flutter/material.dart';

class ProcessedImage extends StatelessWidget {
  final String? base64Image;

  const ProcessedImage({super.key, this.base64Image});

  @override
  Widget build(BuildContext context) {
    if (base64Image == null) return const SizedBox.shrink();

    final imageBytes = base64Decode(base64Image!);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Imagen Procesada:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.memory(
                imageBytes,
                fit: BoxFit.contain,
                width: double.infinity,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
