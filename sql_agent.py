from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage
import re

class SQLChatAgent:
    def __init__(self, llm, connection_string, memory=None):
        print(f"Initializing SQL Agent with connection string: {connection_string}")  # Debug print
        self.connection_string = connection_string
        self.llm = llm
        self.db = SQLDatabase.from_uri(self.connection_string)
        self.memory = memory if memory else ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def extract_sql_query(self, text):
        """Extract SQL query from text that contains markdown code blocks."""
        # Look for SQL between ```sql and ``` markers
        sql_match = re.search(r'```sql\n(.*?)\n```', text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # If no SQL block found, look for any code block
        code_match = re.search(r'```(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        return None
        
    def query(self, user_input):
        """Process a user query and return the response."""
        try:
            # Get the table info for context
            table_info = self.get_table_info()
            print(f"Retrieved table info: {table_info}")  # Debug print
            
            # Create a context-rich message
            prompt = f"""Given this database schema:
            {table_info}
            
            User Question: {user_input}
            
            Please help me with this database query. Remember to:
            1. Include the SQL query you generate (in a ```sql code block)
            2. Present results in a clear format
            3. Provide brief explanations
            4. Ask for clarification if needed
            5. Explain if the question cannot be answered with the schema"""
            
            # Get the response using the LLM directly
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content
            print(f"LLM Response: {response_content}")  # Debug print
            
            # Extract and execute SQL query if present
            sql_query = self.extract_sql_query(response_content)
            if sql_query:
                print(f"Extracted SQL query: {sql_query}")  # Debug print
                try:
                    # Execute the query
                    print(f"Executing query against database...")  # Debug print
                    query_result = self.db.run(sql_query)
                    print(f"Query result: {query_result}")  # Debug print
                    
                    # Add the results to the response
                    if query_result is not None:  # Changed condition to check for None
                        response_content += f"\n\nQuery Results:\n{query_result}"
                    else:
                        response_content += "\n\nQuery executed successfully but returned no results."
                except Exception as e:
                    print(f"Error executing query: {str(e)}")  # Debug print
                    response_content += f"\n\nError executing query: {str(e)}"
            else:
                print("No SQL query found in LLM response")  # Debug print
            
            # Add messages to memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response_content)
            
            return response_content
            
        except Exception as e:
            error_msg = f"Error processing your query: {str(e)}"
            print(f"SQL Agent Error: {str(e)}")  # Debug print
            return error_msg
    
    def get_table_info(self):
        """Get information about tables in the database."""
        try:
            info = self.db.get_table_info()
            print(f"Retrieved table info: {info}")  # Debug print
            return info
        except Exception as e:
            print(f"Error getting table info: {str(e)}")  # Debug print
            return str(e)
    
    def get_sample_rows(self, table_name, limit=5):
        """Get sample rows from a specific table."""
        try:
            sample_query = f"SELECT * FROM {table_name} LIMIT {limit}"
            print(f"Executing sample query: {sample_query}")  # Debug print
            result = self.db.run(sample_query)
            print(f"Sample query result: {result}")  # Debug print
            return result
        except Exception as e:
            print(f"Error getting sample rows: {str(e)}")  # Debug print
            return f"Error retrieving sample rows: {str(e)}" 