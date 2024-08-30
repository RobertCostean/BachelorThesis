import sqlite3
import os

# Define the database path
db_path = 'instance/users.db'  # Replace with the correct path

# Print the absolute path to verify
print("Using database file at:", os.path.abspath(db_path))

# Connect to the existing database
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Verify the existence of the table
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()
print("Existing tables:", tables)

try:
    # Add the new column 'confirmed' to the 'user' table
    c.execute('ALTER TABLE user ADD COLUMN confirmed BOOLEAN DEFAULT 0')
    print("Column 'investment' added successfully.")
except sqlite3.OperationalError as e:
    print("Error occurred:", e)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database schema update script executed.")
