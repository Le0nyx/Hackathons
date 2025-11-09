import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/screens/done_quests.dart';
import 'package:convert_to_green/screens/entry_screen.dart';
import 'package:convert_to_green/screens/leaderboard_screen.dart';
import 'package:convert_to_green/screens/login_screen.dart';
import 'package:convert_to_green/screens/quest_screen.dart';
import 'package:convert_to_green/screens/purchase_history_screen.dart';
import 'package:convert_to_green/screens/shop_screen.dart';
import 'package:flutter/material.dart';

class WBDrawer extends StatelessWidget {
  const WBDrawer({super.key});

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: SafeArea(
        child: Column(
          children: [
            kDefaultSpacer,
            Text("Menu", style: TextTheme.of(context).titleLarge),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: kLargeSpacing * 2,
              ).copyWith(top: kDefaultSpacing),
              child: const Divider(),
            ),
            ListTile(
              title: Text("Home"),
              leading: Icon(Icons.home),
              onTap: () {
                Navigator.of(
                  context,
                ).push(MaterialPageRoute(builder: (context) => EntryScreen()));
              },
            ),
            ListTile(
              title: Text("Quests"),
              leading: Icon(Icons.list),
              onTap: () {
                Navigator.of(
                  context,
                ).push(MaterialPageRoute(builder: (context) => QuestScreen()));
              },
            ),
            ListTile(
              title: Text("My Progress"),
              leading: Icon(Icons.person),
              onTap: () {
                Navigator.of(
                  context,
                ).push(MaterialPageRoute(builder: (context) => DoneQuests()));
              },
            ),
            ListTile(
              title: Text("Leaderboard"),
              leading: Icon(Icons.leaderboard),
              onTap: () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (context) => LeaderboardScreen()),
                );
              },
            ),
            ListTile(
              title: Text("Shop"),
              leading: Icon(Icons.shopping_cart),
              onTap: () {
                Navigator.of(
                  context,
                ).push(MaterialPageRoute(builder: (context) => ShopScreen()));
              },
            ),
            ListTile(
              title: Text("Purchase History"),
              leading: Icon(Icons.receipt),
              onTap: () {
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => PurchaseHistoryScreen(),
                  ),
                );
              },
            ),
            const Spacer(),
            ListTile(
              title: Text(
                "Logout",
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
              leading: Icon(
                Icons.logout,
                color: Theme.of(context).colorScheme.error,
              ),
              onTap: () {
                logout(context);
              },
            ),
          ],
        ),
      ),
    );
  }

  void logout(BuildContext context) {
    Navigator.of(
      context,
    ).push(MaterialPageRoute(builder: (context) => LoginScreen()));
  }
}
