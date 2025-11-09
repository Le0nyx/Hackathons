import 'dart:convert';
import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/widgets/base/app_bar.dart';
import 'package:convert_to_green/widgets/base/drawer.dart.dart';
import 'package:convert_to_green/widgets/done_quests/user_card.dart';
import 'package:convert_to_green/widgets/main_screen/quest_card.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class DoneQuests extends StatefulWidget {
  const DoneQuests({super.key});

  @override
  State<DoneQuests> createState() => _DoneQuestsState();
}

class _DoneQuestsState extends State<DoneQuests> {
  late Future<Map<String, dynamic>> _userDataFuture;

  @override
  void initState() {
    super.initState();
    _userDataFuture = fetchUserData();
  }

  /// Fetch user details and completed quests
  Future<Map<String, dynamic>> fetchUserData() async {
    final prefs = await SharedPreferences.getInstance();
    final int? userId = prefs.getInt("id");

    if (userId == null) throw Exception("User ID not found");

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

    // Fetch completed quests
    final questsUrl = Uri.parse(
      'http://localhost:3000/get/quests/user?userId=$userId',
    );
    final questsResponse = await http.get(
      questsUrl,
      headers: {'Content-Type': 'application/json'},
    );
    if (questsResponse.statusCode != 200) {
      throw Exception('Failed to fetch completed quests');
    }
    final questsData = jsonDecode(questsResponse.body);

    return {'user': userData, 'quests': questsData};
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: WBAppBar(
        title: const Text("My Progress"),
        leading: Builder(
          builder: (context) => IconButton(
            icon: const Icon(Icons.menu),
            onPressed: () => Scaffold.of(context).openDrawer(),
          ),
        ),
      ),
      drawer: const WBDrawer(),
      body: FutureBuilder<Map<String, dynamic>>(
        future: _userDataFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData) {
            return const Center(child: Text('No data found.'));
          }

          final user = snapshot.data!['user'];
          final quests = snapshot.data!['quests'] as List<dynamic>;

          final totalToday = quests.length.toString();

          return SingleChildScrollView(
            padding: const EdgeInsets.all(kDefaultSpacing),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                UserCard(user: user, showProgress: true),
                kSmallSpacer,
                const Divider(),
                kDefaultSpacer,
                Card(
                  color: Colors.green,
                  child: Padding(
                    padding: const EdgeInsets.all(kDefaultSpacing),
                    child: Row(
                      children: [
                        const Text("Today's Achievements"),
                        const Spacer(),
                        Text("$totalToday"),
                      ],
                    ),
                  ),
                ),
                kDefaultSpacer,
                Text(
                  "Your completed quests:",
                  style: Theme.of(
                    context,
                  ).textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: kSmallSpacing),
                Column(
                  children: quests.map((quest) {
                    return Padding(
                      padding: const EdgeInsets.symmetric(
                        vertical: kSmallSpacing,
                      ),
                      child: QuestCard(
                        title: quest['Title'] ?? 'Untitled Quest',
                        description: quest['Description'] ?? '',
                        gold: quest['Value']?.toString() ?? '0',
                        questId: quest['Id'],
                        refreshCallback: () {
                          setState(() {
                            _userDataFuture = fetchUserData();
                          });
                        },
                        showDialogs: false,
                      ),
                    );
                  }).toList(),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}
