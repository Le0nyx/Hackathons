import 'package:flutter/material.dart';

class WBAppBar extends AppBar {
  WBAppBar({
    super.key,
    super.title,
    super.actions,
    super.backgroundColor,
    Widget? leading,
  }) : super(
         automaticallyImplyLeading: false,
         leading: leading == null
             ? null
             : SizedBox(key: leadingKey, child: leading),
       );

  static const leadingKey = ValueKey('WBAppBar_leading_wrapper_key');
}
