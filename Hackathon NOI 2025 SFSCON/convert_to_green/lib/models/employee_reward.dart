class EmployeeReward {
  final int id;
  final String employeeName;
  final String rewardTitle;
  final String rewardDescription;
  final int coinsRequired;

  EmployeeReward({
    required this.id,
    required this.employeeName,
    required this.rewardTitle,
    required this.rewardDescription,
    required this.coinsRequired,
  });

  factory EmployeeReward.fromJson(Map<String, dynamic> json) {
    return EmployeeReward(
      id: json['Id'],
      employeeName: json['EmployeeName'],
      rewardTitle: json['RewardTitle'],
      rewardDescription: json['RewardDescription'],
      coinsRequired: json['CoinsRequired'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'Id': id,
      'EmployeeName': employeeName,
      'RewardTitle': rewardTitle,
      'RewardDescription': rewardDescription,
      'CoinsRequired': coinsRequired,
    };
  }
}
