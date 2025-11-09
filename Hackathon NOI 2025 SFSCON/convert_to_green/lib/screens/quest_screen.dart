import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'package:convert_to_green/widgets/base/app_bar.dart';
import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/widgets/base/drawer.dart.dart';
import 'package:convert_to_green/widgets/main_screen/quest_card.dart';
import 'package:shared_preferences/shared_preferences.dart';

class QuestScreen extends StatefulWidget {
  const QuestScreen({super.key});

  @override
  State<QuestScreen> createState() => _QuestScreenState();
}

class _QuestScreenState extends State<QuestScreen> {
  late Future<List<dynamic>> _questsFuture;

  String _selectedCategory = 'COMMUTE';

  @override
  void initState() {
    super.initState();
    _questsFuture = fetchQuests();
  }

  Future<List<dynamic>> fetchQuests() async {
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    final int? userId = prefs.getInt("id");
    final url = Uri.parse('http://localhost:3000/get/quests?id=$userId');

    final response = await http.get(
      url,
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data;
    } else {
      throw Exception('Failed to load quests');
    }
  }

  Map<String, List<dynamic>> groupByCategory(List<dynamic> quests) {
    final Map<String, List<dynamic>> grouped = {};
    for (var quest in quests) {
      final category = quest['Category'] ?? 'Uncategorized';
      if (!grouped.containsKey(category)) {
        grouped[category] = [];
      }
      grouped[category]!.add(quest);
    }
    return grouped;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: WBAppBar(
        title: const Text("My Quests"),
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
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: FutureBuilder<List<dynamic>>(
          future: _questsFuture,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            } else if (snapshot.hasError) {
              return Center(child: Text('Error: ${snapshot.error}'));
            } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
              return const Center(child: Text('No quests available'));
            }

            final quests = snapshot.data!;
            final groupedQuests = groupByCategory(quests);
            final filteredQuests = groupedQuests[_selectedCategory] ?? [];

            final children = <Widget>[];

            children.add(
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  spacing: kDefaultSpacing,
                  children: [
                    ...groupedQuests.keys.map((category) {
                      return ChoiceChip(
                        label: Text(category),
                        selected: _selectedCategory == category,
                        showCheckmark: false,
                        onSelected: (selected) {
                          setState(() {
                            print(category);
                            _selectedCategory = category;
                          });
                        },
                      );
                    }),
                  ],
                ),
              ),
            );

            children.addAll(
              filteredQuests.map((entry) {
                return QuestCard(
                  title: entry['Title'] ?? 'Untitled Quest',
                  description: entry['Description'] ?? '',
                  gold: entry['Value']?.toString() ?? '0',
                  questId: entry['Id'],
                  refreshCallback: () {
                    setState(() {
                      _questsFuture = fetchQuests();
                    });
                  },
                );
              }).toList(),
            );
            return SingleChildScrollView(
              child: Column(
                spacing: kDefaultSpacing,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: children,
              ),
            );
          },
        ),
      ),
    );
  }
}
