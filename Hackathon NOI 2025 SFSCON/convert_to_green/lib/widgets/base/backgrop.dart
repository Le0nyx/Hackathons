import 'package:flutter/material.dart';

import '../../../../constants.dart';

class Backdrop extends StatelessWidget {
  const Backdrop({required this.child, super.key});

  final Widget child;

  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: Colors.transparent,
    body: ColoredBox(
      color: Theme.of(context).colorScheme.shadow.withValues(alpha: 0.9),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: kDialogWidth),
          child: child,
        ),
      ),
    ),
  );
}
