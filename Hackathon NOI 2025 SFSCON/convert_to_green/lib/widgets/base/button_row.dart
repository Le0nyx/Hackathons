import 'package:flutter/widgets.dart';

import '../../../../constants.dart';

class ButtonRow extends StatelessWidget {
  const ButtonRow({
    required this.primaryButton,
    super.key,
    this.secondaryButton,
  });

  final Widget primaryButton;
  final Widget? secondaryButton;

  @override
  Widget build(BuildContext context) => Row(
    spacing: kLargeSpacing,
    children: [
      if (secondaryButton != null) Expanded(child: secondaryButton!),
      Expanded(flex: 2, child: primaryButton),
    ],
  );
}
