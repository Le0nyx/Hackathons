// ignore_for_file: avoid_classes_with_only_static_members

import 'package:flutter/material.dart';

import '../../constants.dart';
import 'theme_base.dart';
import 'theme_util.dart';

// http://material-foundation.github.io/material-theme-builder/?primary=%2392BF2E&custom%3ALight+Blue=%23ABD2ED&custom%3AOrange=%23FFBC42&custom%3ARed=%23D6573B&bodyFont=Poppins&displayFont=Poppins&colorMatch=true

ThemeData get theme {
  final baseTextTheme = createTextTheme('Poppins', 'Poppins');
  final textTheme = baseTextTheme.copyWith(
    bodyLarge: baseTextTheme.bodyLarge,
    bodyMedium: baseTextTheme.bodyMedium,
    bodySmall: baseTextTheme.bodySmall,
    labelLarge: baseTextTheme.labelLarge,
    labelMedium: baseTextTheme.labelMedium,
    labelSmall: baseTextTheme.labelSmall,
    headlineLarge: baseTextTheme.headlineLarge!.copyWith(
      fontWeight: FontWeight.bold,
    ),
    headlineMedium: baseTextTheme.headlineMedium!.copyWith(
      fontWeight: FontWeight.bold,
    ),
    headlineSmall: baseTextTheme.headlineSmall!.copyWith(
      fontWeight: FontWeight.bold,
    ),
    titleLarge: baseTextTheme.titleLarge!.copyWith(fontWeight: FontWeight.bold),
    titleMedium: baseTextTheme.titleMedium!.copyWith(
      fontWeight: FontWeight.bold,
    ),
    titleSmall: baseTextTheme.titleSmall!.copyWith(fontWeight: FontWeight.bold),
    displayLarge: baseTextTheme.displayLarge!.copyWith(
      fontWeight: FontWeight.bold,
    ),
    displayMedium: baseTextTheme.displayMedium!.copyWith(
      fontWeight: FontWeight.bold,
    ),
    displaySmall: baseTextTheme.displaySmall!.copyWith(
      fontWeight: FontWeight.bold,
    ),
  );

  final baseTheme = MaterialTheme(textTheme).light();
  final backgroundColor = baseTheme.colorScheme.surfaceContainerLowest;
  final shadowColor = baseTheme.splashColor;
  const elevation = 15.0;
  final disabledColor = baseTheme.colorScheme.surfaceContainerHighest;

  return baseTheme.copyWith(
    colorScheme: baseTheme.colorScheme.copyWith(
      onPrimaryContainer: baseTheme.colorScheme.surfaceContainerLowest,
    ),
    visualDensity: VisualDensity.standard,
    // Added to make ChoiceChips denser on Terminal
    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    scaffoldBackgroundColor: backgroundColor,
    appBarTheme: baseTheme.appBarTheme.copyWith(
      centerTitle: true,
      backgroundColor: baseTheme.colorScheme.primaryContainer,
    ),
    splashColor: baseTheme.colorScheme.secondaryContainer,
    disabledColor: disabledColor,
    progressIndicatorTheme: baseTheme.progressIndicatorTheme.copyWith(
      linearMinHeight: 4,
    ),
    cardTheme: baseTheme.cardTheme.copyWith(
      elevation: elevation,
      shadowColor: shadowColor,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadiusGeometry.circular(kDeafultRadius),
      ),
      color: backgroundColor,
    ),
    chipTheme: baseTheme.chipTheme.copyWith(
      selectedColor: baseTheme.colorScheme.primaryContainer,
      backgroundColor: backgroundColor,
      padding: const EdgeInsets.symmetric(
        horizontal: kLargeSpacing,
        vertical: kDefaultSpacing,
      ),
      labelStyle: textTheme.titleMedium!.copyWith(
        color: baseTheme.colorScheme.outlineVariant,
      ),
      secondaryLabelStyle: textTheme.titleMedium!.copyWith(
        color: backgroundColor,
      ),
    ),
    inputDecorationTheme: baseTheme.inputDecorationTheme.copyWith(
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(kDeafultRadius),
        borderSide: BorderSide.none,
      ),
      fillColor: backgroundColor,
      filled: true,
      labelStyle: TextStyle(color: baseTheme.colorScheme.outline),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: backgroundColor,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadiusGeometry.circular(kDeafultRadius),
        ),
        padding: const EdgeInsets.symmetric(
          horizontal: kLargeSpacing,
          vertical: kDefaultSpacing,
        ),
        shadowColor: shadowColor,
        elevation: elevation,
        disabledBackgroundColor: disabledColor,
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        backgroundColor: backgroundColor,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadiusGeometry.circular(kDeafultRadius),
        ),
        padding: const EdgeInsets.symmetric(
          horizontal: kLargeSpacing,
          vertical: kDefaultSpacing,
        ),
        shadowColor: shadowColor,
        elevation: elevation,
        disabledBackgroundColor: disabledColor,
      ),
    ),
    dialogTheme: baseTheme.dialogTheme.copyWith(
      barrierColor: Colors.transparent,
      backgroundColor: backgroundColor,
      insetPadding: const EdgeInsets.all(kSmallSpacing),
    ),
  );
}
