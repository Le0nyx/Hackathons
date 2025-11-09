import 'dart:convert';
import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/widgets/base/app_bar.dart';
import 'package:convert_to_green/widgets/base/buttons.dart';
import 'package:convert_to_green/widgets/base/drawer.dart.dart';
import 'package:convert_to_green/widgets/leaderboard/leaderboard_card.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class LeaderboardScreen extends StatefulWidget {
  const LeaderboardScreen({super.key});

  @override
  State<LeaderboardScreen> createState() => _LeaderboardScreenState();
}

class _LeaderboardScreenState extends State<LeaderboardScreen> {
  bool _onlyToday = true;
  late TextEditingController _searchController;

  @override
  void initState() {
    super.initState();
    _searchController = TextEditingController();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  /// Fetch leaderboard data from API
  Future<List<dynamic>> fetchLeaderboard() async {
    final endoint = _onlyToday ? '/leaderboard' : '/leaderboard/alltime';
    final url = Uri.parse('http://localhost:3000$endoint');
    final response = await http.get(
      url,
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);

      // Optional: sort by Coins descending
      data.sort((a, b) => (b['Coins'] as int).compareTo(a['Coins'] as int));

      return data;
    } else {
      throw Exception('Failed to load leaderboard');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: WBAppBar(
        title: const Text('Leaderboard'),
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
      drawer: WBDrawer(),
      body: FutureBuilder<List<dynamic>>(
        future: fetchLeaderboard(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return const Center(child: Text('No leaderboard data available.'));
          }

          final leaderboardData = snapshot.data!;

          return SingleChildScrollView(
            child: Column(
              children: [
                kDefaultSpacer,
                // Filter buttons
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ChoiceChip(
                      label: const Text("Today"),
                      selected: _onlyToday == true,
                      onSelected: (_) {
                        setState(() {
                          _onlyToday = true;
                        });
                      },
                      showCheckmark: false,
                    ),
                    const SizedBox(width: kDefaultSpacing),
                    ChoiceChip(
                      label: const Text("All Time"),
                      selected: _onlyToday == false,
                      onSelected: (_) {
                        setState(() {
                          _onlyToday = false;
                        });
                      },
                      showCheckmark: false,
                    ),
                  ],
                ),

                // Search bar
                Padding(
                  padding: const EdgeInsets.all(kDefaultSpacing),
                  child: TextFormField(
                    controller: _searchController,
                    decoration: const InputDecoration(
                      labelText: 'Search Users',
                      prefixIcon: Icon(Icons.search),
                    ),
                  ),
                ),
                WBTextButton(
                  onPressed: () {
                    setState(() {});
                  },
                  label: "Search",
                ),
                kDefaultSpacer,
                const Divider(),
                // Leaderboard entries
                ...leaderboardData.asMap().entries.map((entry) {
                  final index = entry.key;
                  final item = entry.value;

                  final placement = index + 1;
                  final firstName = item['Name'] ?? 'Unknown';
                  final lastName = item['Surname'] ?? '';
                  final department = item['Department'] ?? '';
                  final level = item['Level'] ?? 0;
                  final coins = item['Coins'] ?? 0;

                  return LeaderboardCard(
                    fristName: firstName,
                    lastName: lastName,
                    department: department,
                    level: level,
                    placement: placement,
                    coins: coins,
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
