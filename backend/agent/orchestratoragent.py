import os
import asyncio
import logging
import json
from typing import Any, Dict

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from dataclasses import dataclass

# Import your enhanced Databricks plugin
from plugins.databricks_plugin import DatabricksPlugin

# Configure logging
logging.basicConfig(level=logging.INFO)

@dataclass
class AgentConfig:
    azure_endpoint: str
    azure_deployment: str
    azure_api_key: str
    model: str

    @classmethod
    def from_env(cls):
        return cls(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        )

class AzureFoundryAgent:
    """
    Enhanced AI Agent that acts as an orchestrator for Databricks queries with better function calling.
    """
    def __init__(self, cfg: AgentConfig):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create kernel
        self.kernel = sk.Kernel()
        
        # Register the AI service
        self.chat_completion = AzureChatCompletion(
            service_id=cfg.azure_deployment,
            deployment_name=cfg.azure_deployment,
            endpoint=cfg.azure_endpoint,
            api_key=cfg.azure_api_key
        )
        self.kernel.add_service(self.chat_completion)

        # Register Custom Plugins
        self.kernel.add_plugin(DatabricksPlugin(), plugin_name="DatabricksPlugin")
        self.logger.info("Enhanced DatabricksPlugin registered with the kernel.")
        
        # Initialize chat history
        self.chat_history = ChatHistory()

    async def run(self, goal: str) -> Dict[str, Any]:
        """
        Executes a goal using enhanced function calling for Databricks queries.
        """
        try:
            # Enhanced system message with better instructions
            system_message = """You are a helpful AI assistant specialized in data analysis using Databricks. You have access to several functions:

1. list_tables - Lists all available tables in the database
2. describe_table - Shows the structure and columns of a specific table
3. execute_sql_query - Executes SQL queries and returns formatted results
4. execute_query_for_chart - Executes queries and formats results for chart visualization

IMPORTANT GUIDELINES:
- Always explore available tables using list_tables if the user asks about data without specifying a table
- Always describe table structure using describe_table before writing complex queries
- Write efficient SQL queries with appropriate LIMIT clauses
- For visualization requests, use execute_query_for_chart function
- Provide clear explanations of your analysis
- Handle errors gracefully and suggest solutions

When users ask about data, follow this pattern:
1. Understand what they want to know
2. Check available tables if needed
3. Examine table structure if needed
4. Write and execute appropriate SQL
5. Interpret results for the user"""
            
            # Clear chat history and add system message
            self.chat_history.clear()
            self.chat_history.add_system_message(system_message)
            self.chat_history.add_user_message(goal)
            
            # Try to get execution settings for function calling
            try:
                # Try the newer approach first
                from semantic_kernel.connectors.ai.open_ai import OpenAIPromptExecutionSettings
                from semantic_kernel.functions import FunctionChoiceBehavior
                
                execution_settings = OpenAIPromptExecutionSettings(
                    service_id=self.chat_completion.service_id,
                    max_tokens=3000,
                    temperature=0.1,
                    function_choice_behavior=FunctionChoiceBehavior.Auto()
                )
                
                self.logger.info("Using OpenAI function calling with Auto behavior")
                
            except ImportError:
                # Fallback approaches
                try:
                    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
                    execution_settings = PromptExecutionSettings(
                        service_id=self.chat_completion.service_id,
                        max_tokens=3000,
                        temperature=0.1
                    )
                    self.logger.info("Using fallback execution settings")
                except ImportError:
                    execution_settings = None
                    self.logger.warning("No execution settings available")
            
            # Execute the chat with function calling
            if execution_settings:
                response = await self.chat_completion.get_chat_message_contents(
                    chat_history=self.chat_history,
                    settings=execution_settings,
                    kernel=self.kernel
                )
            else:
                # Basic call without settings
                response = await self.chat_completion.get_chat_message_contents(
                    chat_history=self.chat_history,
                    kernel=self.kernel
                )
            
            # Process response
            if response:
                result_text = str(response[0])
                self.chat_history.add_assistant_message(result_text)
                
                # Check if response contains chart data
                graph_data = None
                if "graph_data" in result_text or "chart" in goal.lower() or "graph" in goal.lower():
                    graph_data = self._extract_chart_data(result_text)
                
                return {
                    "goal": goal,
                    "result": result_text,
                    "graph_data": graph_data,
                    "status": "success"
                }
            else:
                return {
                    "goal": goal,
                    "result": "No response received from the model",
                    "status": "error"
                }

        except Exception as e:
            self.logger.error(f"Error during execution: {e}")
            
            # Enhanced fallback with intelligent query detection
            try:
                return await self._intelligent_fallback(goal)
                
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    "goal": goal,
                    "result": f"I encountered an error: {str(e)}. This might be due to function calling configuration. Please check your Semantic Kernel version and Azure OpenAI setup.",
                    "status": "error"
                }

    async def _intelligent_fallback(self, goal: str) -> Dict[str, Any]:
        """Enhanced fallback that tries to understand user intent and execute appropriate functions."""
        goal_lower = goal.lower()
        plugin = self.kernel.plugins.get("DatabricksPlugin")
        
        if not plugin:
            raise Exception("DatabricksPlugin not found")
        
        # Intent detection and appropriate function calls
        if any(word in goal_lower for word in ["list", "show", "available", "tables"]):
            # User wants to see available tables
            result = await plugin["list_tables"].invoke(self.kernel)
            return {
                "goal": goal,
                "result": f"Here are the available tables:\n{result}",
                "status": "success"
            }
            
        elif any(word in goal_lower for word in ["describe", "structure", "columns", "schema"]):
            # User wants table structure
            # Try to extract table name
            words = goal_lower.split()
            table_name = None
            for word in words:
                if word not in ["describe", "table", "structure", "columns", "schema", "of", "the", "what", "is"]:
                    table_name = word
                    break
            
            if table_name:
                result = await plugin["describe_table"].invoke(self.kernel, table_name=table_name)
                return {
                    "goal": goal,
                    "result": f"Table structure for {table_name}:\n{result}",
                    "status": "success"
                }
        
        elif any(word in goal_lower for word in ["count", "total", "number"]):
            # User wants count data
            if "telemetry_data" in goal_lower:
                sql_query = "SELECT COUNT(*) as total_records FROM telemetry_data"
                result = await plugin["execute_sql_query"].invoke(self.kernel, query=sql_query)
                return {
                    "goal": goal,
                    "result": f"I executed this query to get the count: {sql_query}\n\nResult:\n{result}",
                    "status": "success"
                }
        
        elif any(word in goal_lower for word in ["chart", "graph", "visualize", "plot"]):
            # User wants visualization
            # This would need more sophisticated query generation
            result = "For chart creation, I need a specific SQL query. Please specify what data you want to visualize and I'll help create the appropriate query."
            return {
                "goal": goal,
                "result": result,
                "status": "success"
            }
        
        # Default response
        return {
            "goal": goal,
            "result": "I understand you want to work with Databricks data. Could you please be more specific? I can help you:\n- List available tables\n- Describe table structures\n- Execute SQL queries\n- Create visualizations",
            "status": "success"
        }

    def _extract_chart_data(self, response_text: str) -> dict:
        """Extract chart data from response text."""
        try:
            # Look for JSON data in the response
            import re
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
                if "graph_data" in data:
                    return data["graph_data"]
            
            # Look for direct JSON in response
            try:
                if response_text.strip().startswith('{'):
                    data = json.loads(response_text)
                    if "graph_data" in data:
                        return data["graph_data"]
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"Could not extract chart data: {e}")
        
        return None

async def _async_main():
    """Enhanced main entry point for testing."""
    try:
        cfg = AgentConfig.from_env()
        if not all([cfg.azure_endpoint, cfg.azure_api_key, cfg.azure_deployment]):
            raise ValueError("Missing one or more required environment variables.")

        agent = AzureFoundryAgent(cfg)

        # Test various types of queries
        test_queries = [
            "What tables are available in the database?",
            "Describe the structure of the telemetry_data table",
            "What is the total number of users in the 'telemetry_data' table?",
            "Show me the top 10 users by activity from telemetry_data",
        ]

        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Testing query: {query}")
            print('='*50)
            result = await agent.run(query)
            print(f"Status: {result['status']}")
            print(f"Result: {result['result']}")
            if result.get('graph_data'):
                print(f"Chart data available: {bool(result['graph_data'])}")

    except Exception as e:
        print(f"Failed to run the agent: {e}")

if __name__ == "__main__":
    asyncio.run(_async_main())