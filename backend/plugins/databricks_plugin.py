import os
import json
from databricks import sql
from typing import List, Dict, Any, Optional

# Import Semantic Kernel types for function registration
from semantic_kernel.functions import kernel_function

class DatabricksPlugin:
    """
    Enhanced plugin for connecting to Databricks and executing SQL queries with better error handling and metadata support.
    """
    def __init__(self):
        self._host = os.environ.get("DATABRICKS_HOST")
        self._token = os.environ.get("DATABRICKS_TOKEN")
        self._http_path = os.environ.get("DATABRICKS_HTTP_PATH")
        self._catalog = os.environ.get("DATABRICKS_CATALOG", "default")
        self._schema = os.environ.get("DATABRICKS_SCHEMA", "default")

    def _get_connection(self):
        """Get a Databricks connection with proper error handling."""
        if not all([self._host, self._token, self._http_path]):
            raise ValueError("Missing Databricks credentials in environment variables. Please set DATABRICKS_HOST, DATABRICKS_TOKEN, and DATABRICKS_HTTP_PATH.")
        
        return sql.connect(
            server_hostname=self._host,
            http_path=self._http_path,
            access_token=self._token
        )

    @kernel_function(
        description="Lists all available tables in the specified catalog and schema. Useful for understanding what data is available.",
        name="list_tables",
    )
    async def list_tables(self, catalog: str = None, schema: str = None) -> str:
        """
        Lists all tables in the specified catalog and schema.
        
        Args:
            catalog: The catalog name (optional, uses default if not provided)
            schema: The schema name (optional, uses default if not provided)
            
        Returns:
            Formatted string with available tables
        """
        catalog = catalog or self._catalog
        schema = schema or self._schema
        
        try:
            with self._get_connection() as connection:
                with connection.cursor() as cursor:
                    query = f"SHOW TABLES IN {catalog}.{schema}"
                    cursor.execute(query)
                    tables = cursor.fetchall()
                    
                    if not tables:
                        return f"No tables found in {catalog}.{schema}"
                    
                    result = f"Available tables in {catalog}.{schema}:\n"
                    for table in tables:
                        # Table info usually comes as (database, tableName, isTemporary)
                        table_name = table[1] if len(table) > 1 else str(table[0])
                        result += f"- {table_name}\n"
                    
                    return result
                    
        except Exception as e:
            return f"Error listing tables: {e}"

    @kernel_function(
        description="Describes the structure of a specific table including column names and data types. Essential before querying a table.",
        name="describe_table",
    )
    async def describe_table(self, table_name: str, catalog: str = None, schema: str = None) -> str:
        """
        Describes the structure of a table.
        
        Args:
            table_name: Name of the table to describe
            catalog: The catalog name (optional)
            schema: The schema name (optional)
            
        Returns:
            Formatted string with table structure
        """
        catalog = catalog or self._catalog
        schema = schema or self._schema
        full_table_name = f"{catalog}.{schema}.{table_name}"
        
        try:
            with self._get_connection() as connection:
                with connection.cursor() as cursor:
                    query = f"DESCRIBE TABLE {full_table_name}"
                    cursor.execute(query)
                    columns = cursor.fetchall()
                    
                    if not columns:
                        return f"Table {full_table_name} not found or has no columns"
                    
                    result = f"Table structure for {full_table_name}:\n"
                    result += "Column Name | Data Type | Comment\n"
                    result += "-" * 50 + "\n"
                    
                    for col in columns:
                        col_name = col[0]
                        col_type = col[1]
                        col_comment = col[2] if len(col) > 2 else ""
                        result += f"{col_name} | {col_type} | {col_comment}\n"
                    
                    return result
                    
        except Exception as e:
            return f"Error describing table {table_name}: {e}"

    @kernel_function(
        description="Executes a given SQL query on Databricks and returns the results. Use this for data analysis and retrieval.",
        name="execute_sql_query",
    )
    async def execute_sql_query(self, query: str, limit: int = 100) -> str:
        """
        Connects to Databricks and executes a SQL query with enhanced formatting.
        
        Args:
            query: The SQL query to execute
            limit: Maximum number of rows to return (default: 100)
            
        Returns:
            Formatted string with query results
        """
        try:
            # Add limit to query if not present and it's a SELECT statement
            query_upper = query.upper().strip()
            if query_upper.startswith('SELECT') and 'LIMIT' not in query_upper:
                query = f"{query.rstrip(';')} LIMIT {limit}"
            
            with self._get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    if not rows:
                        return "Query executed successfully, but no results were returned."
                    
                    # Get column names
                    column_names = [desc[0] for desc in cursor.description]
                    
                    # Format results
                    result_str = f"Query Results ({len(rows)} rows):\n"
                    
                    # Create header
                    header = " | ".join(f"{col:15}" for col in column_names)
                    result_str += header + "\n"
                    result_str += "-" * len(header) + "\n"
                    
                    # Add data rows
                    for row in rows:
                        formatted_row = " | ".join(f"{str(cell):15}" for cell in row)
                        result_str += formatted_row + "\n"
                    
                    # Add summary if many rows
                    if len(rows) >= limit:
                        result_str += f"\n(Showing first {limit} rows. Use LIMIT in your query to see more.)"
                    
                    return result_str
                    
        except Exception as e:
            error_msg = str(e)
            if "TABLE_OR_VIEW_NOT_FOUND" in error_msg:
                return f"Table not found. Error: {error_msg}\n\nTip: Use the list_tables function to see available tables, or describe_table to check table structure."
            elif "COLUMN_NOT_FOUND" in error_msg:
                return f"Column not found. Error: {error_msg}\n\nTip: Use describe_table function to see available columns."
            else:
                return f"An error occurred while executing the query: {error_msg}"

    @kernel_function(
        description="Executes a SQL query and returns results in JSON format suitable for creating charts and graphs.",
        name="execute_query_for_chart",
    )
    async def execute_query_for_chart(self, query: str, chart_type: str = "bar") -> str:
        """
        Executes a query and formats results for chart creation.
        
        Args:
            query: SQL query to execute
            chart_type: Type of chart (bar, line, pie)
            
        Returns:
            JSON string with chart-ready data
        """
        try:
            with self._get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    
                    if not rows:
                        return json.dumps({"error": "No data returned from query"})
                    
                    # Convert to list of dictionaries
                    data = []
                    for row in rows:
                        row_dict = dict(zip(column_names, row))
                        data.append(row_dict)
                    
                    chart_data = {
                        "type": chart_type,
                        "data": {
                            "labels": [str(row[column_names[0]]) for row in rows],
                            "datasets": [{
                                "label": column_names[1] if len(column_names) > 1 else "Value",
                                "data": [row[column_names[1]] if len(column_names) > 1 else row[column_names[0]] for row in rows],
                                "backgroundColor": [
                                    "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", 
                                    "#9966FF", "#FF9F40", "#FF6384", "#C9CBCF"
                                ]
                            }]
                        },
                        "options": {
                            "responsive": True,
                            "plugins": {
                                "title": {
                                    "display": True,
                                    "text": "Query Results"
                                }
                            }
                        }
                    }
                    
                    return json.dumps({
                        "graph_data": chart_data,
                        "raw_data": data
                    })
                    
        except Exception as e:
            return json.dumps({"error": f"Error executing query for chart: {e}"})
