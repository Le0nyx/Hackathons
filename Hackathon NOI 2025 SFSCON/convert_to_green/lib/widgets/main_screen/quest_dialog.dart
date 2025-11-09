import 'dart:convert';

import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/widgets/base/app_bar.dart';
import 'package:convert_to_green/widgets/base/backgrop.dart';
import 'package:convert_to_green/widgets/base/button_row.dart';
import 'package:convert_to_green/widgets/base/buttons.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class QuestDialog extends StatelessWidget {
  const QuestDialog({
    super.key,
    required this.title,
    required this.description,
    required this.gold,
    required this.questId,
    this.refreshCallback,
  });

  final String title;
  final String description;
  final String gold;
  final int questId;
  final VoidCallback? refreshCallback;

  @override
  Widget build(BuildContext context) {
    return Backdrop(
      child: Dialog(
        clipBehavior: Clip.hardEdge,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            WBAppBar(title: Text("Quest")),
            kLargeSpacer,
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: kLargeSpacing * 2,
              ),
              child: Text(
                title,
                style: Theme.of(
                  context,
                ).textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.bold),
              ),
            ),
            kDefaultSpacer,
            if (description.isNotEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: kLargeSpacing * 2,
                ),
                child: Text(
                  description,
                  style: Theme.of(
                    context,
                  ).textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.bold),
                ),
              ),
            kLargeSpacer,
            Text(
              "Reward Gold: $gold",
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            Text(
              "Reward XP: $gold",
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            Padding(
              padding: const EdgeInsets.all(kDefaultSpacing),
              child: ButtonRow(
                primaryButton: WBElevatedButton(
                  label: "Done!",
                  onPressed: () {
                    completeQuest(questId, context);
                  },
                  color: ButtonColor.primary,
                ),
                secondaryButton: WBElevatedButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  label: "Close",
                  color: ButtonColor.error,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> completeQuest(int questId, BuildContext context) async {
    // ⚠️ Change localhost to your computer IP if testing on a device
    final url = Uri.parse(
      'http://localhost:3000/quest/complete',
    ); // For Android emulator

    final SharedPreferences prefs = await SharedPreferences.getInstance();
    final int? userId = prefs.getInt("id");

    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'userId': userId, 'questId': questId}),
    );

    if (!context.mounted) return;

    if (response.statusCode == 200) {
      if (refreshCallback != null) {
        refreshCallback!();
      }
      Navigator.of(context).pop(); // Close the dialog
    } else {
      // Handle error
    }
  }
}
