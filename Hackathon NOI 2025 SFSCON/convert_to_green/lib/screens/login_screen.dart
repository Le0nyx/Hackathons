import 'dart:convert';

import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/screens/entry_screen.dart';
import 'package:convert_to_green/widgets/base/app_bar.dart';
import 'package:convert_to_green/widgets/base/buttons.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/svg.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  late TextEditingController _usernameController;
  late TextEditingController _passwordController;

  @override
  void initState() {
    super.initState();
    _usernameController = TextEditingController();
    _passwordController = TextEditingController();
  }

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: WBAppBar(title: Text("Login")),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: kLargeSpacing * 2),
        child: SingleChildScrollView(
          child: Column(
            children: [
              Hero(
                tag: 'logo',
                child: SvgPicture.asset('assets/logo.svg', height: 250),
              ),
              Padding(
                padding: const EdgeInsets.all(kDefaultSpacing),
                child: TextFormField(
                  controller: _usernameController,
                  decoration: InputDecoration(
                    labelText: "Username",
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(50),
                    ),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(kDefaultSpacing),
                child: TextFormField(
                  controller: _passwordController,
                  decoration: InputDecoration(
                    labelText: "Password",
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(50),
                    ),
                  ),
                  obscureText: true,
                ),
              ),
              kLargeSpacer,
              SizedBox(
                width: 150,
                child: WBElevatedButton(
                  onPressed: () async {
                    final username = _usernameController.text;
                    final password = _passwordController.text;

                    var url = Uri.parse('http://localhost:3000/login');
                    var response = await http.post(
                      url,
                      headers: {'Content-Type': 'application/json'},
                      body: jsonEncode({
                        'username': username,
                        'password': password,
                      }),
                    );

                    if (!context.mounted) return;

                    if (response.statusCode != 200) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text('Login failed: ${response.body}  '),
                        ),
                      );
                      return;
                    }
                    final SharedPreferences prefs =
                        await SharedPreferences.getInstance();

                    final Map<String, dynamic> data = jsonDecode(response.body);

                    // Access the employee ID
                    final int id = data['employee']['id'];
                    prefs.setInt("id", id);
                    if (!context.mounted) return;

                    Navigator.of(context).push(
                      MaterialPageRoute(builder: (context) => EntryScreen()),
                    );
                  },
                  label: "Login",
                  color: ButtonColor.primary,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
