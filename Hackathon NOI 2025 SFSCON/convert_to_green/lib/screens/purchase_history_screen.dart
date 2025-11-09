import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class PurchaseHistoryScreen extends StatefulWidget {
  const PurchaseHistoryScreen({super.key});

  @override
  State<PurchaseHistoryScreen> createState() => _PurchaseHistoryScreenState();
}

class _PurchaseHistoryScreenState extends State<PurchaseHistoryScreen> {
  List<Purchase> _purchases = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadPurchases();
  }

  Future<void> _loadPurchases() async {
    setState(() => _isLoading = true);

    try {
      final prefs = await SharedPreferences.getInstance();
      final int? userId = prefs.getInt("id");

      if (userId == null) {
        throw Exception('User not logged in');
      }

      final url = Uri.parse(
        'http://localhost:3000/get/employee_rewards?userId=$userId',
      );
      final response = await http.get(
        url,
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        setState(() {
          _purchases = data.map((json) => Purchase.fromJson(json)).toList();
          _isLoading = false;
        });
      } else {
        throw Exception('Failed to load purchase history');
      }
    } catch (e) {
      setState(() {
        _purchases = [];
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Purchase History')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _purchases.isEmpty
          ? const Center(child: Text('No purchases yet'))
          : ListView.builder(
              padding: const EdgeInsets.all(8),
              itemCount: _purchases.length,
              itemBuilder: (context, index) =>
                  _buildPurchaseCard(_purchases[index]),
            ),
    );
  }

  Widget _buildPurchaseCard(Purchase purchase) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: ListTile(
        contentPadding: const EdgeInsets.all(16),
        title: Text(
          purchase.productName,
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Text('${purchase.description}\n${_formatDate(purchase.date)}'),
        ),
        trailing: Text(
          '${purchase.amount} coins',
          style: Theme.of(
            context,
          ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
        ),
        isThreeLine: true,
      ),
    );
  }

  String _formatDate(DateTime date) {
    return '${date.day.toString().padLeft(2, '0')}/${date.month.toString().padLeft(2, '0')}/${date.year}';
  }
}

class Purchase {
  final String id;
  final String productName;
  final String description;
  final DateTime date;
  final int amount;

  Purchase({
    required this.id,
    required this.productName,
    required this.description,
    required this.date,
    required this.amount,
  });

  factory Purchase.fromJson(Map<String, dynamic> json) {
    return Purchase(
      id: json['Id'].toString(),
      productName: json['Title'] ?? 'Unknown Product',
      description: json['Description'] ?? '',
      date: DateTime.parse(json['Bought']),
      amount: json['CoinsRequired'] ?? 0,
    );
  }
}
