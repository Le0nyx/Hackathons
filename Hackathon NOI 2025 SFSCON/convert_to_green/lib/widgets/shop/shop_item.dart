import 'dart:convert';
import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/widgets/base/buttons.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class ShopItem extends StatefulWidget {
  const ShopItem({
    super.key,
    required this.id,
    required this.title,
    required this.description,
    required this.cost,
    this.imageUrl,
    this.onPurchaseComplete,
  });

  final int id;
  final String title;
  final String description;
  final int cost;
  final String? imageUrl;
  final VoidCallback? onPurchaseComplete; // Callback to refresh parent

  @override
  State<ShopItem> createState() => _ShopItemState();
}

class _ShopItemState extends State<ShopItem> {
  bool _isPurchasing = false;

  /// Handle the purchase of this item
  Future<void> _buyItem() async {
    setState(() => _isPurchasing = true);

    try {
      final SharedPreferences prefs = await SharedPreferences.getInstance();
      final int? userId = prefs.getInt("id");

      if (userId == null) {
        throw Exception("User not logged in");
      }

      final url = Uri.parse('http://localhost:3000/buy');
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'userId': userId, 'rewardId': widget.id}),
      );

      if (!mounted) return;

      if (response.statusCode == 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('${widget.title} purchased successfully!'),
            backgroundColor: Colors.green,
          ),
        );

        // Update user's coin balance in SharedPreferences
        final int currentCoins = prefs.getInt("coins") ?? 0;
        await prefs.setInt("coins", currentCoins - widget.cost);

        // Notify parent to refresh
        if (widget.onPurchaseComplete != null) {
          widget.onPurchaseComplete!();
        }
      } else {
        // Error from backend
        final errorData = jsonDecode(response.body);
        final errorMessage = errorData['error'] ?? 'Purchase failed';

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(errorMessage), backgroundColor: Colors.red),
        );
      }
    } catch (e) {
      // Network or other errors
      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: ${e.toString()}'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _isPurchasing = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: kDefaultSpacing),
      child: Card(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(kDeafultRadius),
          side: BorderSide(
            color: Theme.of(context).colorScheme.outlineVariant,
            width: 2,
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(kDefaultSpacing),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          widget.title,
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 4),
                        Text("Price: ${widget.cost} coins"),
                      ],
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 8),
              Text(widget.description),

              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerRight,
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 150),
                  child: WBElevatedButton(
                    onPressed: _isPurchasing ? null : _buyItem,
                    label: _isPurchasing ? "Buying..." : "Buy",
                    icon: Icons.shopping_cart,
                    color: ButtonColor.primary,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
