import 'dart:convert';
import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/widgets/shop/shop_item.dart';
import 'package:convert_to_green/widgets/base/app_bar.dart';
import 'package:convert_to_green/widgets/base/drawer.dart.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class ShopScreen extends StatefulWidget {
  const ShopScreen({super.key});

  @override
  State<ShopScreen> createState() => _ShopScreenState();
}

class _ShopScreenState extends State<ShopScreen> {
  Key _refreshKey = UniqueKey();

  void _refreshShop() {
    setState(() {
      _refreshKey = UniqueKey();
    });
  }

  /// Fetch rewards and user data from backend
  Future<Map<String, dynamic>> fetchShopData() async {
    final prefs = await SharedPreferences.getInstance();
    final int? userId = prefs.getInt("id");

    // Fetch user info
    final userUrl = Uri.parse(
      'http://localhost:3000/get/employee?userId=$userId',
    );
    final userResponse = await http.get(
      userUrl,
      headers: {'Content-Type': 'application/json'},
    );
    if (userResponse.statusCode != 200) {
      throw Exception('Failed to fetch user data');
    }
    final userData = jsonDecode(userResponse.body);

    // Fetch rewards
    final url = Uri.parse('http://localhost:3000/get/rewards');
    final response = await http.get(
      url,
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return {'rewards': data, 'coins': userData['Coins'] ?? 0};
    } else {
      throw Exception('Failed to load rewards');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: WBAppBar(
        title: const Text("Shop"),
        leading: Builder(
          builder: (context) {
            return IconButton(
              icon: const Icon(Icons.menu),
              onPressed: () {
                Scaffold.of(context).openDrawer();
              },
            );
          },
        ),
      ),

      drawer: const WBDrawer(),
      body: FutureBuilder<Map<String, dynamic>>(
        key: _refreshKey,
        future: fetchShopData(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData) {
            return const Center(child: Text('No data available.'));
          }

          final shopData = snapshot.data!;
          final rewards = shopData['rewards'] as List<dynamic>;
          final int coins = shopData['coins'] ?? 0;

          if (rewards.isEmpty) {
            return const Center(child: Text('No rewards available.'));
          }

          return SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                kDefaultSpacer,
                // Coin Balance Display
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: kDefaultSpacing,
                    vertical: kSmallSpacing,
                  ),
                  child: Text(
                    'My Balance: $coins Coins',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                ...rewards.map((reward) {
                  return Padding(
                    padding: const EdgeInsets.symmetric(
                      horizontal: kDefaultSpacing,
                      vertical: kSmallSpacing,
                    ),
                    child: ShopItem(
                      id: reward['Id'],
                      title: reward['Title'] ?? 'Untitled',
                      description: reward['Description'] ?? '',
                      cost: reward['CoinsRequired'] ?? 0,
                      imageUrl: reward['ImageUrl'] ?? '',
                      onPurchaseComplete: _refreshShop,
                    ),
                  );
                }),
              ],
            ),
          );
        },
      ),
    );
  }
}
