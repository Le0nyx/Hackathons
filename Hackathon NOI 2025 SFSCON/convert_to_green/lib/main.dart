import 'package:convert_to_green/screens/login_screen.dart';
import 'package:flutter/material.dart';
import 'widgets/base/theme.dart' as app_theme;

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: LoginScreen(),
      theme: app_theme.theme,
      debugShowCheckedModeBanner: false,
    );
  }
}
