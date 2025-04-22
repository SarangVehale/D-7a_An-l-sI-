import pandas as pd
import sqlite3
import os

def excel_to_sqlite():
    # Ask user for Excel file path
    excel_file = input("Enter the path to your Excel file (e.g., data.xlsx): ").strip()
    if not os.path.exists(excel_file):
        print("❌ File not found. Please check the path and try again.")
        return

    # Ask user for output SQLite DB file name
    db_file = input("Enter the name for the output SQLite DB file (e.g., output.db): ").strip()
    if not db_file.endswith(".db"):
        db_file += ".db"

    # Load the Excel file
    xls = pd.ExcelFile(excel_file)

    # Connect to SQLite database
    conn = sqlite3.connect(db_file)

    # Convert each sheet to a table
    for sheet_name in xls.sheet_names:
        print(f"📄 Processing sheet: {sheet_name}")
        df = xls.parse(sheet_name)
        df.to_sql(sheet_name, conn, if_exists='replace', index=False)
        print(f"✅ Inserted {len(df)} rows into table '{sheet_name}'")

    # Close connection
    conn.close()
    print(f"\n🎉 Conversion complete! Database saved as '{db_file}'")

# Run the function
if __name__ == "__main__":
    excel_to_sqlite()
