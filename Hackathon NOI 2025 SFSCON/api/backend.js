const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

// Database connection with error handling
const db = mysql.createConnection({
    host: 'localhost',
    user: 'myuser',
    password: 'mypassword',
    database: 'mydb'
});

db.connect(err => {
    if (err) {
        console.error('Database connection failed:', err);
        process.exit(1);
    }
    console.log('Connected to database');
});

// Generic error handler
function handleError(res, err) {
    console.error(err);
    res.status(500).send('Server error');
}

// ---- Employees ----

// Register user
app.post('/register', (req, res) => {
    const { username, password } = req.body;
    db.query('INSERT INTO Employees (Username, Password) VALUES (?, ?)', [username, password], (err, results) => {
        if (err) return handleError(res, err);
        res.json({ id: results.insertId, username });
    });
});

// Modify user password
app.put('/Employees/:id', (req, res) => {
    const { password } = req.body;
    db.query('UPDATE Employees SET Password=? WHERE Id=?', [password, req.params.id], err => {
        if (err) return handleError(res, err);
        res.send('User password updated');
    });
});

// Remove user
app.delete('/delete/Employees/:id', (req, res) => {
    db.query('DELETE FROM Employees WHERE Id=?', [req.params.id], err => {
        if (err) return handleError(res, err);
        res.send('User deleted');
    });
});

// Login
app.post('/login', (req, res) => {
    const { username, password } = req.body;
    db.query('SELECT * FROM Employees WHERE Username=? AND Password=?', [username, password], (err, results) => {
        if (err) return handleError(res, err);
        if (!results || results.length === 0) return res.status(401).send('Invalid username or password');
        const user = results[0];
        const employee = {
            id: user.Id,
            name: user.Name,
            surname: user.Surname,
            department: user.Department,
            level: user.Level,
            coins: user.Coins,
            username: user.Username
        };
        res.json({ employee });
    });
});

// ---- EMPLOYEES ----

// Add employee
app.post('/add/employees', (req, res) => {
    const { name, surname, department, level, coins = 0, username, password } = req.body;
    db.query(
        'INSERT INTO Employees (Name, Surname, Department, Level, Coins, Username, Password) VALUES (?, ?, ?, ?, ?, ?, ?)',
        [name, surname, department, level, coins, username, password],
        (err, results) => {
            if (err) return handleError(res, err);
            res.json({ id: results.insertId });
        }
    );
});

// Update employee
app.put('/update/employees/:id', (req, res) => {
    const { name, surname, department, level, coins, username, password } = req.body;
    db.query(
        'UPDATE Employees SET Name=?, Surname=?, Department=?, Level=?, Coins=?, Username=?, Password=? WHERE Id=?',
        [name, surname, department, level, coins, username, password, req.params.id],
        err => {
            if (err) return handleError(res, err);
            res.send('Employee updated');
        }
    );
});

// Delete employee
app.delete('/delete/employees/:id', (req, res) => {
    db.query('DELETE FROM Employees WHERE Id=?', [req.params.id], err => {
        if (err) return handleError(res, err);
        res.send('Employee deleted');
    });
});

app.get('/get/employees', (req, res) => {
    db.query('SELECT * FROM Employees', (err, results) => {
        if (err) return handleError(res, err);
        res.json(results);
    });
});

app.get('/get/employee', (req, res) => {
    const employeeId = req.query.userId;
    console.log('Fetching employee with ID:', employeeId);
    db.query('SELECT * FROM Employees WHERE Id=?', [employeeId], (err, results) => {
        if (err) return handleError(res, err);
        res.json(results[0]);
    });
});

// ---- QUESTS ----

// Add quest
app.post('/add/quests', (req, res) => {
    const { title, value, description } = req.body;
    db.query('INSERT INTO Quests (Title, Value, Description) VALUES (?, ?, ?)', [title, value, description], (err, results) => {
        if (err) return handleError(res, err);
        res.json({ id: results.insertId });
    });
});

// Update quest
app.put('/update/quests/:id', (req, res) => {
    const { title, value, description } = req.body;
    db.query('UPDATE Quests SET Title=?, Value=?, Description=? WHERE Id=?', [title, value, description, req.params.id], err => {
        if (err) return handleError(res, err);
        res.send('Quest updated');
    });
});

// Delete quest
app.delete('/delete/quests/:id', (req, res) => {
    db.query('DELETE FROM Quests WHERE Id=?', [req.params.id], err => {
        if (err) return handleError(res, err);
        res.send('Quest deleted');
    });
});

// Get quests
app.get('/get/quests', (req, res) => {
    const employeeId = req.query.id;
    if (!employeeId) return res.status(400).send('Missing id query parameter');

    const sql = `
        SELECT q.Id, q.Title, q.Value, c.Name AS Category, q.Description
        FROM Quests q
        LEFT JOIN QuestsCategories c ON q.CategoryId = c.Id
        WHERE q.Id NOT IN (
            SELECT QuestId FROM EmployeeQuests WHERE EmployeeId = ?
        )
    `;
    db.query(sql, [employeeId], (err, results) => {
        if (err) return handleError(res, err);
        res.json(results);
    });
});

app.get('/get/quests/user', (req, res) => {
    const userId = req.query.userId;
    if (!userId) return res.status(400).json({ error: 'Missing userId' });

    const query = 'SELECT * FROM EmployeeQuests JOIN Quests ON EmployeeQuests.QuestId = Quests.Id JOIN Employees ON EmployeeQuests.EmployeeId = Employees.Id WHERE Employees.Id = ?';

    db.query(query, [userId], (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(results);
    });
});


// Add questCategory
app.post('/add/questCategory', (req, res) => {
    const { name, description } = req.body;
    db.query('INSERT INTO QuestsCategories (Name, Description) VALUES (?, ?)', [name, description], (err, results) => {
        if (err) return handleError(res, err);
        res.json({ id: results.insertId });
    });
});

// ---- REWARDS ----

// Add reward
app.post('/add/rewards', (req, res) => {
    const { employeeId, title, description, coinsRequired } = req.body;
    db.query('INSERT INTO Rewards (Id, Title, Description, CoinsRequired) VALUES (?, ?, ?, ?)',
        [Id, title, description, coinsRequired], (err, results) => {
            if (err) return handleError(res, err);
            res.json({ id: results.insertId });
        });
});

// Get rewards
app.get('/get/rewards', (req, res) => {
    db.query('SELECT * FROM Rewards ', (err, results) => {
        if (err) return handleError(res, err);
        res.json(results);
    });
});

// Update reward
app.put('/update/rewards/:id', (req, res) => {
    const { title, description, coinsRequired } = req.body;
    db.query('UPDATE Rewards SET Title=?, Description=?, CoinsRequired=? WHERE Id=?',
        [title, description, coinsRequired, req.params.id], err => {
            if (err) return handleError(res, err);
            res.send('Reward updated');
        });
});

// Delete reward
app.delete('/delete/rewards/:id', (req, res) => {
    db.query('DELETE FROM Rewards WHERE Id=?', [req.params.id], err => {
        if (err) return handleError(res, err);
        res.send('Reward deleted');
    });
});

// ---- LEADERBOARD ----

app.get('/leaderboard', (req, res) => {
    const query = `
SELECT 
    e.Id, e.Name, e.Surname, e.Department, e.Level, e.Coins,MAX(qe.Timestamp) AS LatestQuestTimestam FROM Employees e LEFT JOIN EmployeeQuests qe ON e.Id = qe.EmployeeID AND DATE(qe.Timestamp) = CURDATE() GROUP BY  e.Id, e.Name, e.Surname, e.Department, e.Level, e.Coins ORDER BY e.Level DESC;`;
    db.query(query, (err, results) => {
        if (err) return handleError(res, err);
        res.json(results);
    });
});

app.get('/leaderboard/alltime', (req, res) => {
    const query = `SELECT e.Id, e.Name, e.Surname, e.Department, e.Level, e.Coins FROM Employees e ORDER BY e.Level DESC;`;
    db.query(query, (err, results) => {
        if (err) return handleError(res, err);
        res.json(results);
    });
});

// ---- EMPLOYEE REWARDS ----

// Add an employee reward record
app.post('/add/employee_rewards', (req, res) => {
    const { rewardId, employeeId, boughtDateTime } = req.body;
    db.query('INSERT INTO EmployeeRewards (RewardId, EmployeeId, BoughtDateTime) VALUES (?, ?, ?)',
        [rewardId, employeeId, boughtDateTime || new Date()], (err, results) => {
            if (err) return handleError(res, err);
            res.json({ id: results.insertId });
        });
});

// Get all employee rewards
app.get('/get/employee_rewards', (req, res) => {
    const userId = req.query.userId;

    if (!userId) {
        db.query('SELECT * FROM EmployeeRewards JOIN Rewards ON EmployeeRewards.RewardId = Rewards.Id JOIN Employees ON EmployeeRewards.EmployeeId = Employees.Id;', (err, results) => {
            if (err) return handleError(res, err);
            res.json(results);
        });
    } else {
        const query = `
            SELECT 
                EmployeeRewards.Id,
                Rewards.Title,
                Rewards.Description,
                Rewards.CoinsRequired,
                EmployeeRewards.Bought
            FROM EmployeeRewards 
            JOIN Rewards ON EmployeeRewards.RewardId = Rewards.Id 
            WHERE EmployeeRewards.EmployeeId = ?
            ORDER BY EmployeeRewards.Bought DESC
        `;
        db.query(query, [userId], (err, results) => {
            if (err) return handleError(res, err);
            res.json(results);
        });
    }
});

// Delete an employee reward record
app.delete('/delete/employee_rewards/:id', (req, res) => {
    db.query('DELETE FROM EmployeeRewards WHERE Id=?', [req.params.id], err => {
        if (err) return handleError(res, err);
        res.send('Employee reward record deleted');
    });
});


app.post('/quest/complete', (req, res) => {
    const { userId, questId } = req.body;

    if (!userId || !questId) {
        return res.status(400).json({ error: 'Missing userId or questId' });
    }

    const insertQuery = 'INSERT INTO EmployeeQuests (EmployeeID, QuestID, Timestamp) VALUES (?, ?, ?)';

    db.query(insertQuery, [userId, questId, new Date()], err => {
        if (err) return handleError(res, err);
        const updateQuery = `
            UPDATE Employees 
            SET Coins = Coins + (SELECT Value FROM Quests WHERE Id = ?),
                Level = Level + 1
            WHERE Id = ?
        `;
        db.query(updateQuery, [questId, userId], err => {
            if (err) return handleError(res, err);
            res.json({ message: 'Quest completed, coins added, level increased' });
        });
    });
});

app.post('/buy', (req, res) => {
    const { userId, rewardId } = req.body;

    if (!userId || !rewardId) {
        return res.status(400).json({ error: 'Missing userId or rewardId' });
    }

    // Check if user has enough coins
    const checkCoinsQuery = 'SELECT Coins FROM Employees WHERE Id = ?';
    db.query(checkCoinsQuery, [userId], (err, results) => {
        if (err) return handleError(res, err);
        if (results.length === 0) {
            return res.status(404).json({ error: 'User not found' });
        }

        const userCoins = results[0].Coins;

        const getRewardQuery = 'SELECT CoinsRequired FROM Rewards WHERE Id = ?';
        db.query(getRewardQuery, [rewardId], (err, results) => {
            if (err) return handleError(res, err);
            if (results.length === 0) {
                return res.status(404).json({ error: 'Reward not found' });
            }

            const rewardCost = results[0].CoinsRequired;

            if (userCoins < rewardCost) {
                return res.status(400).json({ error: 'Not enough coins' });
            }

            // Deduct coins and add reward to user
            const updateCoinsQuery = 'UPDATE Employees SET Coins = Coins - ? WHERE Id = ?';
            db.query(updateCoinsQuery, [rewardCost, userId], (err) => {
                if (err) return handleError(res, err);

                const addRewardQuery = 'INSERT INTO EmployeeRewards (EmployeeId, RewardId, Bought) VALUES (?, ?, ?)';
                db.query(addRewardQuery, [userId, rewardId, new Date()], (err) => {
                    if (err) return handleError(res, err);
                    res.json({ message: 'Reward purchased successfully' });
                });
            });
        });
    });
});



app.listen(3000, () => {
    console.log('Server running on port 3000');
});
