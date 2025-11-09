class Employee {
  final int? id;
  final String name;
  final String surname;
  final String department;
  final int level;
  final String coins;

  Employee({
    this.id,
    required this.name,
    required this.surname,
    required this.department,
    required this.level,
    required this.coins,
  });

  factory Employee.fromMap(Map<String, dynamic> map) {
    return Employee(
      id: map['id'],
      name: map['name'],
      surname: map['surname'],
      department: map['department'],
      level: map['level'],
      coins: map['coins'],
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'surname': surname,
      'department': department,
      'level': level,
      'coins': coins,
    };
  }
}
