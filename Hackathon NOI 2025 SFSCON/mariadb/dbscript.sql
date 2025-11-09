USE mydb;

CREATE TABLE Employees (
    Id INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(255) NOT NULL,
    Surname VARCHAR(255) NOT NULL,
    Department VARCHAR(255),
    Level INT,
    Coins INT DEFAULT 0,
    Username VARCHAR(255),
    Password VARCHAR(255)
);

CREATE TABLE Quests (
    Id INT PRIMARY KEY AUTO_INCREMENT,
    Title VARCHAR(255) NOT NULL,
    Value INT NOT NULL,
    CategoryId INT,
    Description TEXT,
    FOREIGN KEY (CategoryId) REFERENCES QuestsCategories(Id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE Rewards (
    Id INT PRIMARY KEY AUTO_INCREMENT,
    EmployeeId INT,
    Title VARCHAR(255) NOT NULL,
    Description TEXT,
    CoinsRequired INT,
    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (EmployeeId) REFERENCES Employees(Id)
);

CREATE TABLE EmployeeRewards (
    Id INT PRIMARY KEY AUTO_INCREMENT,
    RewardId INT,
    EmployeeId INT,
    Bought DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (RewardId) REFERENCES Rewards(Id),
    FOREIGN KEY (EmployeeId) REFERENCES Employees(Id)
);

CREATE TABLE EmployeeQuests (
    Id INT PRIMARY KEY AUTO_INCREMENT,
    EmployeeId INT,
    QuestId INT,
    FOREIGN KEY (EmployeeId) REFERENCES Employees(Id),
    FOREIGN KEY (QuestId) REFERENCES Quests(Id)
);

CREATE TABLE QuestsCategories (
    Id INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(50) NOT NULL UNIQUE,
    Description VARCHAR(255)
);