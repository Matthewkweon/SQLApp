import pandas as pd
import sqlite3
import os
import sys
import logging
import re
import json
from datetime import datetime
from tabulate import tabulate
import openai
import tiktoken  # pip install tiktoken
import getpass
from dotenv import load_dotenv

openai_client = None
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("NLPSQLAssistant")

# Global database connection
conn = None
DB_FILE = "assistant_database.db"

# Initialize OpenAI API (will be set later)

def setup_openai():
    global openai_client  # Declare that you’re using the global variable
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found.")
        api_key = getpass.getpass("Enter your OpenAI API key (input will be hidden): ")
    openai.api_key = api_key
    try:
        # Test connection with a simple request
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print("✓ OpenAI API connection successful")
        openai_client = openai  # Assign the module to the global variable
        return True
    except Exception as e:
        print(f"Error connecting to OpenAI API: {e}")
        logger.error(f"OpenAI API connection error: {e}")
        return False


def print_colored(text, color="white"):
    """Print colored text in terminal"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")

def create_connection(db_file=DB_FILE):
    """Create a database connection to a SQLite database"""
    global conn
    try:
        conn = sqlite3.connect(db_file)
        print_colored(f"Connected to SQLite database: {db_file}", "green")
        return True
    except sqlite3.Error as e:
        print_colored(f"Error connecting to database: {e}", "red")
        logger.error(f"Error connecting to database: {e}")
        return False

def close_connection():
    """Close the database connection"""
    global conn
    if conn:
        conn.close()
        conn = None
        print_colored("Database connection closed.", "yellow")

def get_database_schema():
    """Get the complete database schema information"""
    global conn
    if not conn:
        if not create_connection():
            return None
    
    try:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            # Get column information for each table
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            sample_rows = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Store table schema
            schema[table] = {
                "columns": [{"name": col[1], "type": col[2]} for col in columns],
                "sample_data": sample_rows if sample_rows else [],
                "row_count": row_count
            }
        
        return schema
    
    except sqlite3.Error as e:
        print_colored(f"Error retrieving database schema: {e}", "red")
        logger.error(f"Error retrieving database schema: {e}")
        return None

def schema_to_string(schema):
    """Convert schema dictionary to formatted string for LLM prompt"""
    if not schema:
        return "No tables found in the database."
    
    result = []
    for table_name, table_info in schema.items():
        columns_str = ", ".join([f"{col['name']} ({col['type']})" for col in table_info["columns"]])
        result.append(f"Table: {table_name} ({table_info['row_count']} rows)")
        result.append(f"Columns: {columns_str}")
        
        # Add sample data if available
        if table_info["sample_data"]:
            result.append("Sample data:")
            column_names = [col['name'] for col in table_info["columns"]]
            sample_data = []
            for row in table_info["sample_data"]:
                row_data = []
                for i, value in enumerate(row):
                    row_data.append(f"{column_names[i]}: {value}")
                sample_data.append(", ".join(row_data))
            result.append("\n".join(f"- {s}" for s in sample_data))
        
        result.append("")  # Empty line between tables
    
    return "\n".join(result)

def count_tokens(string, model="gpt-3.5-turbo"):
    """Returns the number of tokens in a text string"""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_schema_if_needed(schema_string, max_tokens=3000):
    """Truncate schema information if it exceeds token limit"""
    current_tokens = count_tokens(schema_string)
    
    if current_tokens <= max_tokens:
        return schema_string
    
    # Split by table
    tables = schema_string.split("Table: ")
    
    # Always keep the header (which is empty because split starts with "Table: ")
    result = []
    
    # Add tables until we approach the token limit
    current_count = 0
    for i, table in enumerate(tables[1:], 1):  # Skip first empty element
        table_text = "Table: " + table
        table_tokens = count_tokens(table_text)
        
        if current_count + table_tokens > max_tokens:
            # If this is the first table, we need to include a truncated version
            if i == 1:
                # Just include column names, not sample data
                lines = table_text.split("\n")
                table_header = lines[0]
                columns_line = lines[1]
                truncated_table = f"{table_header}\n{columns_line}\n[Sample data truncated due to length]"
                result.append(truncated_table)
            break
        
        result.append(table_text)
        current_count += table_tokens
    
    # If we couldn't add any complete tables, truncate the schema description
    if not result:
        return schema_string[:int(len(schema_string) * max_tokens / current_tokens)] + "...[truncated]"
    
    return "".join(result)

def generate_sql_from_natural_language(question, schema):
    """Generate SQL from natural language using OpenAI"""
    
    try:
        # Convert schema to string and truncate if needed
        schema_string = schema_to_string(schema)
        schema_string = truncate_schema_if_needed(schema_string)
        
        # Create prompt for the LLM
        prompt = f"""You are an AI assistant tasked with converting user queries into SQL statements. The database uses SQLite and contains the following schema:

{schema_string}

User Query: "{question}"

Your task is to:
1. Generate a SQL query that accurately answers the user's question.
2. Ensure the SQL is compatible with SQLite syntax.
3. Provide a short comment explaining what the query does.

Output Format:
- SQL: [Your SQL query here]
- Explanation: [Your explanation here]

Important:
- If the query cannot be answered with the available schema, explain why.
- Do not make assumptions about data that is not in the schema.
- Use proper SQLite functions and syntax.
- Remember that table and column names are case-sensitive in the query.
"""

        logger.info(f"Sending prompt to OpenAI with token count: {count_tokens(prompt)}")
        
        # Send prompt to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more precise SQL generation
            max_tokens=1000
        )
        
        # Parse and extract response
        response_text = response.choices[0].message.content.strip()
        logger.info(f"Received response from OpenAI: {response_text}")
        
        # Extract SQL and explanation
        sql_match = re.search(r"SQL:?\s+(.*?)(?:\n\n|\n- |\nExplanation:|\Z)", response_text, re.DOTALL)
        explanation_match = re.search(r"Explanation:?\s+(.*)", response_text, re.DOTALL)
        
        sql = sql_match.group(1).strip() if sql_match else None
        explanation = explanation_match.group(1).strip() if explanation_match else None
        
        # Clean SQL (remove code block markers if present)
        if sql and (sql.startswith("```") or sql.startswith("``")):
            sql = re.sub(r"^```\w*\n|```$", "", sql, flags=re.MULTILINE).strip()
        
        return sql, explanation
    
    except Exception as e:
        print_colored(f"Error generating SQL from natural language: {e}", "red")
        logger.error(f"OpenAI API error: {e}")
        return None, f"Error: {str(e)}"

def run_query(query, show_results=True):
    """Run an SQL query and display results"""
    global conn
    if not conn:
        if not create_connection():
            return None
    
    try:
        # Check if query is a SELECT query
        is_select = query.strip().lower().startswith("select")
        
        cursor = conn.cursor()
        start_time = datetime.now()
        
        if is_select:
            # Run SELECT query and fetch results
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if show_results:
                if rows:
                    print_colored(f"\nQuery Results ({len(rows)} rows, {duration:.2f} seconds):", "green")
                    print(tabulate(rows, headers=columns, tablefmt="pretty"))
                    
                    # Ask if user wants to export results
                    export = input("Export results to CSV? (y/n): ").lower()
                    if export == 'y':
                        filename = input("Enter filename (default: query_results.csv): ") or "query_results.csv"
                        df = pd.DataFrame(rows, columns=columns)
                        df.to_csv(filename, index=False)
                        print_colored(f"Results exported to {filename}", "green")
                else:
                    print_colored("Query executed successfully, but returned no results.", "yellow")
            
            return {"columns": columns, "rows": rows, "duration": duration}
        else:
            # Run non-SELECT query
            cursor.execute(query)
            conn.commit()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            rows_affected = cursor.rowcount
            
            if show_results:
                print_colored(f"Query executed successfully. Rows affected: {rows_affected} ({duration:.2f} seconds)", "green")
            
            return {"rows_affected": rows_affected, "duration": duration}
    
    except sqlite3.Error as e:
        print_colored(f"SQL Error: {e}", "red")
        logger.error(f"SQL Error: {e}")
        return None

def process_natural_language_query(question, show_sql=True):
    """Process a natural language query and run the generated SQL"""
    # Get database schema
    schema = get_database_schema()
    if not schema:
        print_colored("Could not retrieve database schema.", "red")
        return
    
    print_colored("Generating SQL from your question...", "cyan")
    sql, explanation = generate_sql_from_natural_language(question, schema)
    
    if not sql:
        print_colored(f"Could not generate SQL: {explanation}", "red")
        return
    
    # Show generated SQL if requested
    if show_sql:
        print_colored("\nGenerated SQL:", "cyan")
        print(sql)
        print_colored("\nExplanation:", "cyan")
        print(explanation)
        
        execute = input("\nExecute this query? (y/n): ").lower()
        if execute != 'y':
            print_colored("Query execution cancelled.", "yellow")
            return
    
    # Run the query
    return run_query(sql)

def load_csv(csv_file, table_name=None):
    """Load a CSV file into the database"""
    global conn
    if not conn:
        if not create_connection():
            return False
    
    # Validate CSV file
    if not os.path.exists(csv_file):
        print_colored(f"CSV file does not exist: {csv_file}", "red")
        return False
    
    try:
        # Extract and sanitize table name if not provided
        if table_name is None:
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Sanitize table name
        table_name = re.sub(r'[^\w]', '_', table_name)
        if table_name and table_name[0].isdigit():
            table_name = 'table_' + table_name
        
        print_colored(f"Using table name: {table_name}", "cyan")
        
        # Read CSV into DataFrame
        df = pd.read_csv(csv_file)
        print_colored(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns.", "green")
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            print_colored(f"Table '{table_name}' already exists.", "yellow")
            while True:
                choice = input("Do you want to (1) Replace, (2) Append, or (3) Cancel? [1/2/3]: ")
                if choice == '1':
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    print_colored(f"Table '{table_name}' replaced with new data.", "green")
                    break
                elif choice == '2':
                    df.to_sql(table_name, conn, if_exists='append', index=False)
                    print_colored(f"Data appended to table '{table_name}'.", "green")
                    break
                elif choice == '3':
                    print_colored("CSV import cancelled.", "yellow")
                    return False
                else:
                    print_colored("Invalid choice. Please enter 1, 2, or 3.", "red")
        else:
            # Create new table
            df.to_sql(table_name, conn, if_exists='fail', index=False)
            print_colored(f"Created new table '{table_name}' with CSV data.", "green")
        
        return True
    except Exception as e:
        print_colored(f"Error loading CSV: {e}", "red")
        logger.error(f"Error loading CSV: {e}")
        return False

def list_tables():
    """List all tables in the database"""
    global conn
    if not conn:
        if not create_connection():
            return
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print_colored("No tables found in the database.", "yellow")
            return
        
        print_colored("\nAvailable Tables:", "cyan")
        for i, table in enumerate(tables, 1):
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            row_count = cursor.fetchone()[0]
            print_colored(f"{i}. {table[0]} ({row_count} rows)", "cyan")
    
    except sqlite3.Error as e:
        print_colored(f"Error listing tables: {e}", "red")
        logger.error(f"Error listing tables: {e}")

def describe_table(table_name):
    """Describe a table's schema"""
    global conn
    if not conn:
        if not create_connection():
            return
    
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        if not columns:
            print_colored(f"Table '{table_name}' does not exist.", "red")
            return
        
        print_colored(f"\nSchema for table '{table_name}':", "cyan")
        schema_data = [(col[0], col[1], col[2]) for col in columns]
        print(tabulate(schema_data, headers=["Index", "Column Name", "Data Type"], tablefmt="pretty"))
        
        # Show sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        rows = cursor.fetchall()
        if rows:
            print_colored(f"\nSample data from '{table_name}' (first 5 rows):", "cyan")
            column_names = [col[1] for col in columns]
            print(tabulate(rows, headers=column_names, tablefmt="pretty"))
    
    except sqlite3.Error as e:
        print_colored(f"Error describing table: {e}", "red")
        logger.error(f"Error describing table: {e}")

def display_help():
    """Display help information"""
    print_colored("\n=== Natural Language SQL Assistant Help ===", "magenta")
    print_colored("Available commands:", "cyan")
    print("  ask <natural language question> - Ask a question in plain English")
    print("  load <csv_file> [table_name] - Load a CSV file into the database")
    print("  tables - List all tables in the database")
    print("  describe <table_name> - Show table schema and sample data")
    print("  query <sql_query> - Run an SQL query directly")
    print("  help - Display this help message")
    print("  exit - Exit the assistant")
    print_colored("\nExamples:", "cyan")
    print("  ask Show me the top 5 products by revenue")
    print("  ask What was the total sales in each region last month?")
    print("  load data/sales.csv")
    print("  tables")
    print("  describe sales")
    print_colored("\nTips:", "yellow")
    print("  - Be specific in your questions")
    print("  - You can toggle SQL display with 'show_sql on/off'")
    print("  - SQL queries can span multiple lines (end with semicolon)")
    print("  - Load your CSV data first before asking questions")

def main():
    """Main function for the interactive NLP-SQL assistant"""
    print_colored("\n=== Natural Language SQL Assistant ===", "magenta")
    print_colored("Ask questions about your data in plain English!", "cyan")
    print_colored("Type 'help' for available commands or 'exit' to quit", "cyan")
    
    # Create initial database connection
    create_connection()
    
    # Initial OpenAI setup
    setup_openai()
    
    multi_line_query = []
    collecting_query = False
    show_sql = True
    
    while True:
        try:
            if collecting_query:
                prompt = "... "
                line = input(prompt)
                
                if line.strip().endswith(';'):
                    # End of query
                    multi_line_query.append(line.strip()[:-1])  # Remove semicolon
                    full_query = " ".join(multi_line_query)
                    print_colored(f"Running query: {full_query}", "cyan")
                    run_query(full_query)
                    multi_line_query = []
                    collecting_query = False
                else:
                    multi_line_query.append(line)
            else:
                prompt = "\n> "
                command = input(prompt).strip()
                
                if not command:
                    continue
                
                # Handle the "ask" command (natural language query)
                if command.lower().startswith("ask "):
                    question = command[4:].strip()
                    if question:
                        process_natural_language_query(question, show_sql)
                    else:
                        print_colored("Please provide a question after 'ask'", "red")
                    continue
                
                # Handle show_sql toggle
                if command.lower().startswith("show_sql "):
                    toggle = command[9:].strip().lower()
                    if toggle in ["on", "true", "yes"]:
                        show_sql = True
                        print_colored("SQL display enabled", "green")
                    elif toggle in ["off", "false", "no"]:
                        show_sql = False
                        print_colored("SQL display disabled", "yellow")
                    else:
                        print_colored("Invalid option. Use 'show_sql on' or 'show_sql off'", "red")
                    continue
                
                # Parse other commands
                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd == "exit":
                    break
                elif cmd == "help":
                    display_help()
                elif cmd == "tables":
                    list_tables()
                elif cmd == "describe":
                    if not args:
                        print_colored("Error: Table name required. Use 'describe <table_name>'", "red")
                    else:
                        describe_table(args)
                elif cmd == "load":
                    if not args:
                        print_colored("Error: CSV file path required. Use 'load <csv_file> [table_name]'", "red")
                    else:
                        load_args = args.split(maxsplit=1)
                        csv_file = load_args[0]
                        table_name = load_args[1] if len(load_args) > 1 else None
                        load_csv(csv_file, table_name)
                elif cmd == "query":
                    if not args:
                        print_colored("Enter your SQL query (end with semicolon ';'):", "cyan")
                        collecting_query = True
                    elif args.strip().endswith(';'):
                        # Single line query ending with semicolon
                        run_query(args.strip()[:-1])  # Remove semicolon
                    else:
                        # Start collecting multi-line query
                        multi_line_query = [args]
                        collecting_query = True
                else:
                    # If it doesn't match any command, treat it as a natural language query
                    process_natural_language_query(command, show_sql)
                
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit the assistant.")
        except Exception as e:
            print_colored(f"Error: {e}", "red")
            logger.error(f"Unexpected error: {e}")
    
    # Close connection before exiting
    close_connection()
    print_colored("\nThank you for using Natural Language SQL Assistant. Goodbye!", "magenta")

if __name__ == "__main__":
    main()