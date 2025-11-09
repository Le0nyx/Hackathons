import 'package:convert_to_green/constants.dart';
import 'package:convert_to_green/widgets/main_screen/quest_dialog.dart';
import 'package:flutter/material.dart';

class QuestCard extends StatelessWidget {
  const QuestCard({
    super.key,
    required this.title,
    required this.description,
    required this.gold,
    required this.questId,
    this.refreshCallback,
    this.showDialogs = true,
  });

  final String title;
  final String description;
  final String gold;
  final int questId;
  final VoidCallback? refreshCallback;
  final bool showDialogs;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: showDialogs ? () => showQuestDialog(context) : null,
      child: Card(
        color: Theme.of(context).colorScheme.surfaceContainerLow,
        child: Padding(
          padding: const EdgeInsets.all(kDefaultSpacing),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(title, style: TextTheme.of(context).titleMedium),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [Text("Reward Gold:"), Text(gold)],
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [Text("Reward XP:"), Text(gold)],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> showQuestDialog(BuildContext context) async {
    showDialog(
      context: context,
      builder: (context) {
        return QuestDialog(
          title: title,
          description: description,
          gold: gold,
          questId: questId,
          refreshCallback: refreshCallback,
        );
      },
    );
  }
}
