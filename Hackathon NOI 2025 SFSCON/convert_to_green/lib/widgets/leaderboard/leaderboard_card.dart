import 'package:convert_to_green/constants.dart';
import 'package:flutter/material.dart';

class LeaderboardCard extends StatelessWidget {
  const LeaderboardCard({
    super.key,
    required this.fristName,
    required this.lastName,
    required this.department,
    required this.level,
    required this.placement,
    required this.coins,
  });

  final String fristName;
  final String lastName;
  final String department;
  final int level;
  final int placement;
  final int coins;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(
        horizontal: kDefaultSpacing,
        vertical: kDefaultSpacing / 2,
      ),
      child: Padding(
        padding: const EdgeInsets.all(kDefaultSpacing),
        child: Row(
          children: [
            // Placement (Rank number)
            Text(
              "# $placement",
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            kLargeSpacer,

            // User details
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      fristName,
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    kDefaultSpacer,
                    Text(
                      lastName,
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                Row(
                  children: [
                    Text("Level $level"),
                    kLargeSpacer,
                    Text(department),
                  ],
                ),
              ],
            ),

            const Spacer(),

            // Coins / Points
            Padding(
              padding: const EdgeInsets.only(right: kLargeSpacing),
              child: Text(
                "$coins pts",
                style: Theme.of(context).textTheme.headlineSmall,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
