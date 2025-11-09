import 'dart:convert';
import 'package:convert_to_green/screens/quest_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/widgets/base/app_bar.dart';
import 'package:convert_to_green/widgets/base/drawer.dart.dart';

/// Fetch the quests for the current user
Future<List<dynamic>> fetchUserQuests() async {
  final prefs = await SharedPreferences.getInstance();
  final int? userId = prefs.getInt("id");

  if (userId == null) throw Exception("User ID not found");

  // Fetch the user's quests
  final questsUrl = Uri.parse(
    'http://localhost:3000/get/quests/user?userId=$userId',
  );

  final questsResponse = await http.get(
    questsUrl,
    headers: {'Content-Type': 'application/json'},
  );

  if (questsResponse.statusCode != 200) {
    throw Exception('Failed to fetch user quests');
  }

  final questsData = jsonDecode(questsResponse.body);

  return questsData;
}

class EntryScreen extends StatelessWidget {
  const EntryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: WBAppBar(
        title: const Text("Welcome"),
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
        future: fetchUserQuests(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            // Loading state
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            // Error state
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            // No quests state
            return const Center(child: Text('No quests found.'));
          } else {
            // Data fetched successfully
            final quests = snapshot.data!;
            debugPrint('Fetched Quests: $quests');

            return Column(
              children: [
                Hero(
                  tag: 'logo',
                  child: SvgPicture.asset('assets/logo.svg', height: 100),
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      "Convert to ",
                      style: Theme.of(context).textTheme.headlineMedium,
                    ),
                    Text(
                      "Green",
                      style: Theme.of(
                        context,
                      ).textTheme.headlineMedium?.copyWith(color: Colors.green),
                    ),
                  ],
                ),
                kDefaultSpacer,
                const Divider(),
                kDefaultSpacer,
                SfCircularChart(
                  title: ChartTitle(text: 'Your Carbon Footprint'),
                  legend: Legend(
                    isVisible: true,
                    overflowMode: LegendItemOverflowMode.wrap,
                  ),
                  series: <CircularSeries>[
                    PieSeries<CarbonData, String>(
                      pointColorMapper: (CarbonData data, _) =>
                          data.source == 'Saved' ? Colors.green : null,
                      explode: true,
                      explodeIndex: 0,
                      explodeOffset: '15%',
                      dataSource: getCarbonData(quests.length * 4),
                      xValueMapper: (CarbonData data, _) => data.source,
                      yValueMapper: (CarbonData data, _) => data.emission,
                      dataLabelSettings: const DataLabelSettings(
                        isVisible: false,
                      ),
                    ),
                  ],
                ),
                kLargeSpacer,
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: InkWell(
                      onTap: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (context) => QuestScreen(),
                          ),
                        );
                      },
                      child: Card(
                        color: Theme.of(context).colorScheme.secondaryContainer,
                        child: Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                "View Quests",
                                style: TextTheme.of(context).headlineMedium,
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                const Spacer(),
              ],
            );
          }
        },
      ),
    );
  }

  List<CarbonData> getCarbonData(int saved) {
    return [
      CarbonData('Saved', saved.toDouble()),
      CarbonData('Transportation', 35),
      CarbonData('Energy', 25),
      CarbonData('Food', 15),
      CarbonData('Waste', 10),
      CarbonData('Others', 15),
    ];
  }
}

class CarbonData {
  final String source;
  final double emission;
  CarbonData(this.source, this.emission);
}
