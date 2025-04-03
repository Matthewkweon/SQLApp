# Natural Language SQL Assistant

A command-line tool that translates natural language questions into SQL queries, allowing you to interact with your database using plain English.

## Overview
This tool bridges the gap between natural language and SQL, making it easier for non-technical users to query databases without writing SQL code. It leverages OpenAI's GPT-4 model to interpret questions and generate appropriate SQL queries.
Features

Natural Language Queries: Ask questions about your data in plain English
CSV Import: Easily load CSV files into the SQLite database
Interactive Mode: Command-line interface with multiple helpful commands
Query Visualization: Display query results in formatted tables
Export Capability: Export query results to CSV files
Schema Analysis: Automatically analyzes database schema to inform the AI

## Getting Started
Prerequisites

Python 3.6+
OpenAI API key
Required Python packages (see Installation)

Installation

Clone this repository or download the source code
Install required dependencies:
Copypip install pandas sqlite3 tabulate openai tiktoken python-dotenv

Set up your OpenAI API key:

Create a .env file in the same directory as the script with:
CopyOPENAI_API_KEY=your_api_key_here

Or, you'll be prompted to enter your API key when running the application

Usage
Run the application:
```python
nlp_sql_assistant.py
```
Available Commands

```python
ask <question>: Ask a question in natural language
load <csv_file> [table_name]: Load a CSV file into the database
tables: List all tables in the database
describe <table_name>: Show table schema and sample data
query <sql_query>: Run an SQL query directly (can be multi-line, end with semicolon)
show_sql on/off: Toggle SQL display for natural language queries
help: Display help information
exit: Exit the assistant
```

Example Workflow

Load your CSV data:

```
load sales_data.csv
```
Explore available tables:
```
tables
```

View table structure:
```
describe sales_data
```

Ask questions about your data:
```
> ask What were the total sales by region?
> ask Which product had the highest sales in January 2023?
```


### How It Works

The application maintains a SQLite database connection
When you ask a question, it captures the database schema
The schema and your question are sent to OpenAI's GPT-4
GPT-4 generates an appropriate SQL query based on the context
The application executes the query and displays the results
All interactions are logged for reference

### Troubleshooting

If the application fails to connect to the OpenAI API, check your API key
For SQL errors, review the generated SQL or try rephrasing your question
The assistant works best with clear, specific questions

### Limitations

Complex analytical queries might require refinement
Performance depends on the quality of your database schema
Large databases might require optimization for better performance

License
This project is open-source and available under the MIT License.