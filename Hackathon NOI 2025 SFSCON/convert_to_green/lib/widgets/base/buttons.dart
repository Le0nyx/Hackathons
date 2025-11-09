import 'dart:async';

import 'package:flutter/material.dart';

import '../../constants.dart';

enum ButtonColor { neutral, primary, error }

class WBElevatedButton extends StatefulWidget {
  const WBElevatedButton({
    super.key,
    this.label,
    this.icon,
    this.onPressed,
    this.onLongPressed,
    this.color = ButtonColor.neutral,
    this.maxLines,
    this.textOverflow,
    this.padding,
  });

  final String? label;
  final IconData? icon;
  final FutureOr Function()? onPressed;
  final FutureOr Function()? onLongPressed;
  final ButtonColor color;
  final int? maxLines;
  final TextOverflow? textOverflow;
  final EdgeInsetsGeometry? padding;

  @override
  State<WBElevatedButton> createState() => _WBElevatedButtonState();
}

class _WBElevatedButtonState extends State<WBElevatedButton> {
  bool _ongoing = false;

  @override
  Widget build(BuildContext context) {
    final disabled = widget.onPressed == null && widget.onLongPressed == null;

    final backgroundColor = switch (widget.color) {
      ButtonColor.neutral => null,
      ButtonColor.primary => Theme.of(context).colorScheme.primaryContainer,
      ButtonColor.error => Theme.of(context).colorScheme.errorContainer,
    };
    final foregroundColor = switch (widget.color) {
      ButtonColor.neutral => Theme.of(context).colorScheme.outlineVariant,
      ButtonColor.primary => Theme.of(context).colorScheme.onPrimaryContainer,
      ButtonColor.error => Theme.of(context).colorScheme.onErrorContainer,
    };

    final textStyle = Theme.of(
      context,
    ).textTheme.titleLarge!.copyWith(color: foregroundColor);

    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        backgroundColor: backgroundColor,
        foregroundColor: foregroundColor,
        padding: widget.padding,
      ),
      onPressed: disabled ? null : _onPressed,
      onLongPress: disabled ? null : _onLongPressed,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        spacing: kDefaultSpacing,
        children: [
          if (widget.icon != null)
            Icon(
              widget.icon,
              color: textStyle.color,
              size: textStyle.fontSize,
              weight: textStyle.fontWeight!.value.toDouble(),
            ),
          if (widget.label != null)
            Flexible(
              child: Text(
                widget.label!,
                style: textStyle,
                maxLines: widget.maxLines,
                overflow: widget.textOverflow,
              ),
            ),
        ],
      ),
    );
  }

  FutureOr<void> _onPressed() async {
    if (widget.onPressed == null || _ongoing) {
      return;
    }
    try {
      _ongoing = true;
      await widget.onPressed!();
    } finally {
      if (mounted) {
        _ongoing = false;
      }
    }
  }

  FutureOr<void> _onLongPressed() async {
    if (widget.onLongPressed == null || _ongoing) {
      return;
    }
    try {
      setState(() {
        _ongoing = true;
      });
      await widget.onLongPressed!();
    } finally {
      if (mounted) {
        setState(() {
          _ongoing = false;
        });
      }
    }
  }
}

class WBTextButton extends StatefulWidget {
  const WBTextButton({
    super.key,
    this.label,
    this.icon,
    this.onPressed,
    this.onLongPressed,
    this.color = ButtonColor.neutral,
    this.maxLines,
    this.textOverflow,
    this.padding,
  });

  final String? label;
  final IconData? icon;
  final FutureOr Function()? onPressed;
  final FutureOr Function()? onLongPressed;
  final ButtonColor color;
  final int? maxLines;
  final TextOverflow? textOverflow;
  final EdgeInsetsGeometry? padding;

  @override
  State<WBTextButton> createState() => _WBTextButtonState();
}

class _WBTextButtonState extends State<WBTextButton> {
  bool _ongoing = false;

  @override
  Widget build(BuildContext context) {
    final disabled = widget.onPressed == null && widget.onLongPressed == null;

    final foregroundColor = switch (widget.color) {
      ButtonColor.neutral => Theme.of(context).colorScheme.outlineVariant,
      ButtonColor.primary => Theme.of(context).colorScheme.primaryContainer,
      ButtonColor.error => Theme.of(context).colorScheme.errorContainer,
    };

    final textStyle = Theme.of(
      context,
    ).textTheme.titleLarge!.copyWith(color: foregroundColor);

    return TextButton(
      style: TextButton.styleFrom(
        foregroundColor: foregroundColor,
        elevation: disabled ? 0 : null,
        padding: widget.padding,
      ),
      onPressed: disabled ? null : _onPressed,
      onLongPress: disabled ? null : _onLongPressed,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        spacing: kDefaultSpacing,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (widget.icon != null)
            Icon(
              widget.icon,
              color: textStyle.color,
              size: textStyle.fontSize,
              weight: textStyle.fontWeight!.value.toDouble(),
            ),
          if (widget.label != null)
            Flexible(
              child: Text(
                widget.label!,
                style: textStyle,
                maxLines: widget.maxLines,
                overflow: widget.textOverflow,
              ),
            ),
        ],
      ),
    );
  }

  FutureOr<void> _onPressed() async {
    if (widget.onPressed == null || _ongoing) {
      return;
    }
    try {
      _ongoing = true;

      await widget.onPressed!();
    } finally {
      if (mounted) {
        setState(() {
          _ongoing = false;
        });
      }
    }
  }

  FutureOr<void> _onLongPressed() async {
    if (widget.onLongPressed == null || _ongoing) {
      return;
    }
    try {
      _ongoing = true;
      await widget.onLongPressed!();
    } finally {
      if (mounted) {
        _ongoing = false;
      }
    }
  }
}
