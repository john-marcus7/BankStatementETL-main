CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TIMESTAMP NOT NULL,
    description TEXT NOT NULL,
    amount REAL NOT NULL,
    account_name TEXT NOT NULL,
    transaction_type TEXT CHECK(transaction_type IN ('credit', 'debit')) NOT NULL,
    category_name TEXT
); 

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL,
    pattern TEXT NOT NULL,
    account_name TEXT NOT NULL
)