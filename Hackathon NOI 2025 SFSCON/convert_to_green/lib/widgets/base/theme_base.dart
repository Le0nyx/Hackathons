import "package:flutter/material.dart";

class MaterialTheme {
  final TextTheme textTheme;

  const MaterialTheme(this.textTheme);

  static ColorScheme lightScheme() {
    return const ColorScheme(
      brightness: Brightness.light,
      primary: Color(0xff4a6700),
      surfaceTint: Color(0xff4a6700),
      onPrimary: Color(0xffffffff),
      primaryContainer: Color(0xff92bf2e),
      onPrimaryContainer: Color(0xff344a00),
      secondary: Color(0xff52652d),
      onSecondary: Color(0xffffffff),
      secondaryContainer: Color(0xffd2e8a2),
      onSecondaryContainer: Color(0xff566930),
      tertiary: Color(0xff006d3c),
      onTertiary: Color(0xffffffff),
      tertiaryContainer: Color(0xff21c977),
      onTertiaryContainer: Color(0xff004e2a),
      error: Color(0xffba1a1a),
      onError: Color(0xffffffff),
      errorContainer: Color(0xffffdad6),
      onErrorContainer: Color(0xff93000a),
      surface: Color(0xfff9fbea),
      onSurface: Color(0xff1a1d13),
      onSurfaceVariant: Color(0xff444937),
      outline: Color(0xff747965),
      outlineVariant: Color(0xffc4c9b2),
      shadow: Color(0xff000000),
      scrim: Color(0xff000000),
      inverseSurface: Color(0xff2f3227),
      inversePrimary: Color(0xffa7d644),
      primaryFixed: Color(0xffc2f35e),
      onPrimaryFixed: Color(0xff141f00),
      primaryFixedDim: Color(0xffa7d644),
      onPrimaryFixedVariant: Color(0xff374e00),
      secondaryFixed: Color(0xffd5eba5),
      onSecondaryFixed: Color(0xff141f00),
      secondaryFixedDim: Color(0xffb9cf8b),
      onSecondaryFixedVariant: Color(0xff3b4c17),
      tertiaryFixed: Color(0xff67fea5),
      onTertiaryFixed: Color(0xff00210f),
      tertiaryFixedDim: Color(0xff45e08c),
      onTertiaryFixedVariant: Color(0xff00522c),
      surfaceDim: Color(0xffd9dbcc),
      surfaceBright: Color(0xfff9fbea),
      surfaceContainerLowest: Color(0xffffffff),
      surfaceContainerLow: Color(0xfff3f5e5),
      surfaceContainer: Color(0xffedefdf),
      surfaceContainerHigh: Color(0xffe8ead9),
      surfaceContainerHighest: Color(0xffe2e4d4),
    );
  }

  ThemeData light() {
    return theme(lightScheme());
  }

  static ColorScheme lightMediumContrastScheme() {
    return const ColorScheme(
      brightness: Brightness.light,
      primary: Color(0xff2a3c00),
      surfaceTint: Color(0xff4a6700),
      onPrimary: Color(0xffffffff),
      primaryContainer: Color(0xff567700),
      onPrimaryContainer: Color(0xffffffff),
      secondary: Color(0xff2b3b07),
      onSecondary: Color(0xffffffff),
      secondaryContainer: Color(0xff61743a),
      onSecondaryContainer: Color(0xffffffff),
      tertiary: Color(0xff003f21),
      onTertiary: Color(0xffffffff),
      tertiaryContainer: Color(0xff007e47),
      onTertiaryContainer: Color(0xffffffff),
      error: Color(0xff740006),
      onError: Color(0xffffffff),
      errorContainer: Color(0xffcf2c27),
      onErrorContainer: Color(0xffffffff),
      surface: Color(0xfff9fbea),
      onSurface: Color(0xff0f1209),
      onSurfaceVariant: Color(0xff333827),
      outline: Color(0xff4f5542),
      outlineVariant: Color(0xff6a6f5b),
      shadow: Color(0xff000000),
      scrim: Color(0xff000000),
      inverseSurface: Color(0xff2f3227),
      inversePrimary: Color(0xffa7d644),
      primaryFixed: Color(0xff567700),
      onPrimaryFixed: Color(0xffffffff),
      primaryFixedDim: Color(0xff435d00),
      onPrimaryFixedVariant: Color(0xffffffff),
      secondaryFixed: Color(0xff61743a),
      onSecondaryFixed: Color(0xffffffff),
      secondaryFixedDim: Color(0xff495b24),
      onSecondaryFixedVariant: Color(0xffffffff),
      tertiaryFixed: Color(0xff007e47),
      onTertiaryFixed: Color(0xffffffff),
      tertiaryFixedDim: Color(0xff006236),
      onTertiaryFixedVariant: Color(0xffffffff),
      surfaceDim: Color(0xffc6c8b8),
      surfaceBright: Color(0xfff9fbea),
      surfaceContainerLowest: Color(0xffffffff),
      surfaceContainerLow: Color(0xfff3f5e5),
      surfaceContainer: Color(0xffe8ead9),
      surfaceContainerHigh: Color(0xffdcdece),
      surfaceContainerHighest: Color(0xffd1d3c3),
    );
  }

  ThemeData lightMediumContrast() {
    return theme(lightMediumContrastScheme());
  }

  static ColorScheme lightHighContrastScheme() {
    return const ColorScheme(
      brightness: Brightness.light,
      primary: Color(0xff213100),
      surfaceTint: Color(0xff4a6700),
      onPrimary: Color(0xffffffff),
      primaryContainer: Color(0xff395000),
      onPrimaryContainer: Color(0xffffffff),
      secondary: Color(0xff213100),
      onSecondary: Color(0xffffffff),
      secondaryContainer: Color(0xff3e4f19),
      onSecondaryContainer: Color(0xffffffff),
      tertiary: Color(0xff00341a),
      onTertiary: Color(0xffffffff),
      tertiaryContainer: Color(0xff00552e),
      onTertiaryContainer: Color(0xffffffff),
      error: Color(0xff600004),
      onError: Color(0xffffffff),
      errorContainer: Color(0xff98000a),
      onErrorContainer: Color(0xffffffff),
      surface: Color(0xfff9fbea),
      onSurface: Color(0xff000000),
      onSurfaceVariant: Color(0xff000000),
      outline: Color(0xff292e1e),
      outlineVariant: Color(0xff464b39),
      shadow: Color(0xff000000),
      scrim: Color(0xff000000),
      inverseSurface: Color(0xff2f3227),
      inversePrimary: Color(0xffa7d644),
      primaryFixed: Color(0xff395000),
      onPrimaryFixed: Color(0xffffffff),
      primaryFixedDim: Color(0xff273800),
      onPrimaryFixedVariant: Color(0xffffffff),
      secondaryFixed: Color(0xff3e4f19),
      onSecondaryFixed: Color(0xffffffff),
      secondaryFixedDim: Color(0xff283804),
      onSecondaryFixedVariant: Color(0xffffffff),
      tertiaryFixed: Color(0xff00552e),
      onTertiaryFixed: Color(0xffffffff),
      tertiaryFixedDim: Color(0xff003b1e),
      onTertiaryFixedVariant: Color(0xffffffff),
      surfaceDim: Color(0xffb8baab),
      surfaceBright: Color(0xfff9fbea),
      surfaceContainerLowest: Color(0xffffffff),
      surfaceContainerLow: Color(0xfff0f2e2),
      surfaceContainer: Color(0xffe2e4d4),
      surfaceContainerHigh: Color(0xffd4d6c6),
      surfaceContainerHighest: Color(0xffc6c8b8),
    );
  }

  ThemeData lightHighContrast() {
    return theme(lightHighContrastScheme());
  }

  static ColorScheme darkScheme() {
    return const ColorScheme(
      brightness: Brightness.dark,
      primary: Color(0xffacdb49),
      surfaceTint: Color(0xffa7d644),
      onPrimary: Color(0xff253600),
      primaryContainer: Color(0xff92bf2e),
      onPrimaryContainer: Color(0xff344a00),
      secondary: Color(0xffb9cf8b),
      onSecondary: Color(0xff263502),
      secondaryContainer: Color(0xff3d4f19),
      onSecondaryContainer: Color(0xffabc07e),
      tertiary: Color(0xff4ce690),
      onTertiary: Color(0xff00391d),
      tertiaryContainer: Color(0xff21c977),
      onTertiaryContainer: Color(0xff004e2a),
      error: Color(0xffffb4ab),
      onError: Color(0xff690005),
      errorContainer: Color(0xff93000a),
      onErrorContainer: Color(0xffffdad6),
      surface: Color(0xff12140b),
      onSurface: Color(0xffe2e4d4),
      onSurfaceVariant: Color(0xffc4c9b2),
      outline: Color(0xff8e937e),
      outlineVariant: Color(0xff444937),
      shadow: Color(0xff000000),
      scrim: Color(0xff000000),
      inverseSurface: Color(0xffe2e4d4),
      inversePrimary: Color(0xff4a6700),
      primaryFixed: Color(0xffc2f35e),
      onPrimaryFixed: Color(0xff141f00),
      primaryFixedDim: Color(0xffa7d644),
      onPrimaryFixedVariant: Color(0xff374e00),
      secondaryFixed: Color(0xffd5eba5),
      onSecondaryFixed: Color(0xff141f00),
      secondaryFixedDim: Color(0xffb9cf8b),
      onSecondaryFixedVariant: Color(0xff3b4c17),
      tertiaryFixed: Color(0xff67fea5),
      onTertiaryFixed: Color(0xff00210f),
      tertiaryFixedDim: Color(0xff45e08c),
      onTertiaryFixedVariant: Color(0xff00522c),
      surfaceDim: Color(0xff12140b),
      surfaceBright: Color(0xff373a2f),
      surfaceContainerLowest: Color(0xff0c0f07),
      surfaceContainerLow: Color(0xff1a1d13),
      surfaceContainer: Color(0xff1e2117),
      surfaceContainerHigh: Color(0xff282b21),
      surfaceContainerHighest: Color(0xff33362b),
    );
  }

  ThemeData dark() {
    return theme(darkScheme());
  }

  static ColorScheme darkMediumContrastScheme() {
    return const ColorScheme(
      brightness: Brightness.dark,
      primary: Color(0xffbcec58),
      surfaceTint: Color(0xffa7d644),
      onPrimary: Color(0xff1c2a00),
      primaryContainer: Color(0xff92bf2e),
      onPrimaryContainer: Color(0xff1c2900),
      secondary: Color(0xffcfe59f),
      onSecondary: Color(0xff1c2a00),
      secondaryContainer: Color(0xff84985a),
      onSecondaryContainer: Color(0xff000000),
      tertiary: Color(0xff60f7a0),
      onTertiary: Color(0xff002d15),
      tertiaryContainer: Color(0xff21c977),
      onTertiaryContainer: Color(0xff002c15),
      error: Color(0xffffd2cc),
      onError: Color(0xff540003),
      errorContainer: Color(0xffff5449),
      onErrorContainer: Color(0xff000000),
      surface: Color(0xff12140b),
      onSurface: Color(0xffffffff),
      onSurfaceVariant: Color(0xffdadfc7),
      outline: Color(0xffafb49e),
      outlineVariant: Color(0xff8d937d),
      shadow: Color(0xff000000),
      scrim: Color(0xff000000),
      inverseSurface: Color(0xffe2e4d4),
      inversePrimary: Color(0xff384f00),
      primaryFixed: Color(0xffc2f35e),
      onPrimaryFixed: Color(0xff0b1400),
      primaryFixedDim: Color(0xffa7d644),
      onPrimaryFixedVariant: Color(0xff2a3c00),
      secondaryFixed: Color(0xffd5eba5),
      onSecondaryFixed: Color(0xff0b1400),
      secondaryFixedDim: Color(0xffb9cf8b),
      onSecondaryFixedVariant: Color(0xff2b3b07),
      tertiaryFixed: Color(0xff67fea5),
      onTertiaryFixed: Color(0xff001508),
      tertiaryFixedDim: Color(0xff45e08c),
      onTertiaryFixedVariant: Color(0xff003f21),
      surfaceDim: Color(0xff12140b),
      surfaceBright: Color(0xff43463a),
      surfaceContainerLowest: Color(0xff060803),
      surfaceContainerLow: Color(0xff1c1f15),
      surfaceContainer: Color(0xff26291f),
      surfaceContainerHigh: Color(0xff313429),
      surfaceContainerHighest: Color(0xff3c3f34),
    );
  }

  ThemeData darkMediumContrast() {
    return theme(darkMediumContrastScheme());
  }

  static ColorScheme darkHighContrastScheme() {
    return const ColorScheme(
      brightness: Brightness.dark,
      primary: Color(0xffd4ff7c),
      surfaceTint: Color(0xffa7d644),
      onPrimary: Color(0xff000000),
      primaryContainer: Color(0xffa4d240),
      onPrimaryContainer: Color(0xff070d00),
      secondary: Color(0xffe2f9b1),
      onSecondary: Color(0xff000000),
      secondaryContainer: Color(0xffb5cb87),
      onSecondaryContainer: Color(0xff070d00),
      tertiary: Color(0xffbeffcf),
      onTertiary: Color(0xff000000),
      tertiaryContainer: Color(0xff40dc88),
      onTertiaryContainer: Color(0xff000f05),
      error: Color(0xffffece9),
      onError: Color(0xff000000),
      errorContainer: Color(0xffffaea4),
      onErrorContainer: Color(0xff220001),
      surface: Color(0xff12140b),
      onSurface: Color(0xffffffff),
      onSurfaceVariant: Color(0xffffffff),
      outline: Color(0xffedf3da),
      outlineVariant: Color(0xffc0c5ae),
      shadow: Color(0xff000000),
      scrim: Color(0xff000000),
      inverseSurface: Color(0xffe2e4d4),
      inversePrimary: Color(0xff384f00),
      primaryFixed: Color(0xffc2f35e),
      onPrimaryFixed: Color(0xff000000),
      primaryFixedDim: Color(0xffa7d644),
      onPrimaryFixedVariant: Color(0xff0b1400),
      secondaryFixed: Color(0xffd5eba5),
      onSecondaryFixed: Color(0xff000000),
      secondaryFixedDim: Color(0xffb9cf8b),
      onSecondaryFixedVariant: Color(0xff0b1400),
      tertiaryFixed: Color(0xff67fea5),
      onTertiaryFixed: Color(0xff000000),
      tertiaryFixedDim: Color(0xff45e08c),
      onTertiaryFixedVariant: Color(0xff001508),
      surfaceDim: Color(0xff12140b),
      surfaceBright: Color(0xff4e5145),
      surfaceContainerLowest: Color(0xff000000),
      surfaceContainerLow: Color(0xff1e2117),
      surfaceContainer: Color(0xff2f3227),
      surfaceContainerHigh: Color(0xff3a3d31),
      surfaceContainerHighest: Color(0xff45483c),
    );
  }

  ThemeData darkHighContrast() {
    return theme(darkHighContrastScheme());
  }

  ThemeData theme(ColorScheme colorScheme) => ThemeData(
    useMaterial3: true,
    brightness: colorScheme.brightness,
    colorScheme: colorScheme,
    textTheme: textTheme.apply(
      bodyColor: colorScheme.onSurface,
      displayColor: colorScheme.onSurface,
    ),
    scaffoldBackgroundColor: colorScheme.surface,
    canvasColor: colorScheme.surface,
  );

  /// Light Blue
  static const lightBlue = ExtendedColor(
    seed: Color(0xffabd2ed),
    value: Color(0xffa5d4e6),
    light: ColorFamily(
      color: Color(0xff366474),
      onColor: Color(0xffffffff),
      colorContainer: Color(0xffa5d4e6),
      onColorContainer: Color(0xff2d5d6c),
    ),
    lightMediumContrast: ColorFamily(
      color: Color(0xff366474),
      onColor: Color(0xffffffff),
      colorContainer: Color(0xffa5d4e6),
      onColorContainer: Color(0xff2d5d6c),
    ),
    lightHighContrast: ColorFamily(
      color: Color(0xff366474),
      onColor: Color(0xffffffff),
      colorContainer: Color(0xffa5d4e6),
      onColorContainer: Color(0xff2d5d6c),
    ),
    dark: ColorFamily(
      color: Color(0xffc8efff),
      onColor: Color(0xff003543),
      colorContainer: Color(0xffa5d4e6),
      onColorContainer: Color(0xff2d5d6c),
    ),
    darkMediumContrast: ColorFamily(
      color: Color(0xffc8efff),
      onColor: Color(0xff003543),
      colorContainer: Color(0xffa5d4e6),
      onColorContainer: Color(0xff2d5d6c),
    ),
    darkHighContrast: ColorFamily(
      color: Color(0xffc8efff),
      onColor: Color(0xff003543),
      colorContainer: Color(0xffa5d4e6),
      onColorContainer: Color(0xff2d5d6c),
    ),
  );

  /// Orange
  static const orange = ExtendedColor(
    seed: Color(0xffffbc42),
    value: Color(0xffeac541),
    light: ColorFamily(
      color: Color(0xff725c00),
      onColor: Color(0xffffffff),
      colorContainer: Color(0xffeac541),
      onColorContainer: Color(0xff655200),
    ),
    lightMediumContrast: ColorFamily(
      color: Color(0xff725c00),
      onColor: Color(0xffffffff),
      colorContainer: Color(0xffeac541),
      onColorContainer: Color(0xff655200),
    ),
    lightHighContrast: ColorFamily(
      color: Color(0xff725c00),
      onColor: Color(0xffffffff),
      colorContainer: Color(0xffeac541),
      onColorContainer: Color(0xff655200),
    ),
    dark: ColorFamily(
      color: Color(0xffffe289),
      onColor: Color(0xff3c2f00),
      colorContainer: Color(0xffeac541),
      onColorContainer: Color(0xff655200),
    ),
    darkMediumContrast: ColorFamily(
      color: Color(0xffffe289),
      onColor: Color(0xff3c2f00),
      colorContainer: Color(0xffeac541),
      onColorContainer: Color(0xff655200),
    ),
    darkHighContrast: ColorFamily(
      color: Color(0xffffe289),
      onColor: Color(0xff3c2f00),
      colorContainer: Color(0xffeac541),
      onColorContainer: Color(0xff655200),
    ),
  );

  /// Red
  static const red = ExtendedColor(
    seed: Color(0xffd6573b),
    value: Color(0xffd05e0d),
    light: ColorFamily(
      color: Color(0xff9a4100),
      onColor: Color(0xffffffff),
      colorContainer: Color(0xffc15400),
      onColorContainer: Color(0xfffffbff),
    ),
    lightMediumContrast: ColorFamily(
      color: Color(0xff9a4100),
      onColor: Color(0xffffffff),
      colorContainer: Color(0xffc15400),
      onColorContainer: Color(0xfffffbff),
    ),
    lightHighContrast: ColorFamily(
      color: Color(0xff9a4100),
      onColor: Color(0xffffffff),
      colorContainer: Color(0xffc15400),
      onColorContainer: Color(0xfffffbff),
    ),
    dark: ColorFamily(
      color: Color(0xffffb690),
      onColor: Color(0xff552100),
      colorContainer: Color(0xffe66e21),
      onColorContainer: Color(0xff2c0d00),
    ),
    darkMediumContrast: ColorFamily(
      color: Color(0xffffb690),
      onColor: Color(0xff552100),
      colorContainer: Color(0xffe66e21),
      onColorContainer: Color(0xff2c0d00),
    ),
    darkHighContrast: ColorFamily(
      color: Color(0xffffb690),
      onColor: Color(0xff552100),
      colorContainer: Color(0xffe66e21),
      onColorContainer: Color(0xff2c0d00),
    ),
  );

  List<ExtendedColor> get extendedColors => [lightBlue, orange, red];
}

class ExtendedColor {
  final Color seed, value;
  final ColorFamily light;
  final ColorFamily lightHighContrast;
  final ColorFamily lightMediumContrast;
  final ColorFamily dark;
  final ColorFamily darkHighContrast;
  final ColorFamily darkMediumContrast;

  const ExtendedColor({
    required this.seed,
    required this.value,
    required this.light,
    required this.lightHighContrast,
    required this.lightMediumContrast,
    required this.dark,
    required this.darkHighContrast,
    required this.darkMediumContrast,
  });
}

class ColorFamily {
  const ColorFamily({
    required this.color,
    required this.onColor,
    required this.colorContainer,
    required this.onColorContainer,
  });

  final Color color;
  final Color onColor;
  final Color colorContainer;
  final Color onColorContainer;
}
