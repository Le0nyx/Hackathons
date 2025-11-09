class Quest {
  final int id;
  final String title;
  final int value;
  final String category;
  final String description;

  Quest({
    required this.id,
    required this.title,
    required this.value,
    required this.category,
    required this.description,
  });

  factory Quest.fromJson(Map<String, dynamic> json) {
    return Quest(
      id: json['Id'],
      title: json['Title'],
      value: json['Value'],
      category: json['Category'],
      description: json['Description'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'Id': id,
      'Title': title,
      'Value': value,
      'Category': category,
      'Description': description,
    };
  }
}