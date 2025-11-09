import 'package:convert_to_green/constants.dart';
import 'package:flutter/material.dart';

class UserCard extends StatelessWidget {
  const UserCard({super.key, required this.user, this.showProgress = false});

  /// The user data map returned from backend
  final Map<String, dynamic> user;
  final bool showProgress;

  @override
  Widget build(BuildContext context) {
    final fullName = '${user['Name'] ?? ''} ${user['Surname'] ?? ''}';
    final department = user['Department'] ?? '';
    final level = user['Level'] ?? 0;
    final coins = user['Coins'] ?? 0;

    final progress = showProgress ? (level / 10).clamp(0.0, 1.0) : 0.0;

    return Card(
      color: Theme.of(context).colorScheme.surfaceContainerLow,
      child: Padding(
        padding: const EdgeInsets.all(kDefaultSpacing),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text(fullName, style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 4),
            Text(department, style: Theme.of(context).textTheme.titleMedium),
            if (showProgress) ...[
              kSmallSpacer,
              Row(
                children: [
                  Text(
                    "Level: $level",
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  kLargeSpacer,
                  Expanded(
                    child: LinearProgressIndicator(
                      value: progress,
                      minHeight: 5,
                      backgroundColor: Theme.of(
                        context,
                      ).colorScheme.primaryContainer,
                      color: Colors.red,
                    ),
                  ),
                ],
              ),
            ],
            kSmallSpacer,
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                "Coins: $coins",
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
