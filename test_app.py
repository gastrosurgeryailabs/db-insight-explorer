import unittest
from unittest.mock import patch, MagicMock
from sql_agent import SQLChatAgent
from langchain.schema import HumanMessage, AIMessage

class TestSQLChatAgent(unittest.TestCase):
    @patch('sql_agent.create_sql_agent')
    @patch('sql_agent.SQLDatabase')
    def test_sql_agent_creation(self, mock_sql_database_class, mock_create_agent):
        # Create a proper mock for SQLDatabase that will pass type checking
        mock_db = MagicMock()
        mock_db.get_table_info.return_value = "Table1: col1, col2\nTable2: col3, col4"
        mock_db.run.return_value = "Sample data"
        mock_sql_database_class.from_uri.return_value = mock_db
        
        # Create a mock for the agent result
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Test response"}
        mock_create_agent.return_value = mock_agent
        
        # Create a mock for the LLM that implements required abstract methods
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "LLM response"
        mock_llm.generate_prompt.return_value = MagicMock()
        mock_llm.predict.return_value = "Prediction"
        mock_llm.predict_messages.return_value = "Message prediction"
        
        # Create the agent with our mocks
        agent = SQLChatAgent(mock_llm, "sqlite:///test.db", memory=None)
        
        # Assert that the correct methods were called
        mock_sql_database_class.from_uri.assert_called_once_with("sqlite:///test.db")
        mock_create_agent.assert_called_once()
        
        # Test query method
        result = agent.query("Test query")
        self.assertEqual(result, "Test response")
        # Check that invoke was called with input and chat_history
        self.assertEqual(mock_agent.invoke.call_args[0][0]["input"], "Test query")
        self.assertIn("chat_history", mock_agent.invoke.call_args[0][0])
        
        # Test error handling by making invoke throw an exception
        mock_agent.invoke.side_effect = Exception("Test error")
        result = agent.query("Test query")
        self.assertIn("Error processing your query: Test error", result)
        
        # Test table info method
        result = agent.get_table_info()
        self.assertEqual(result, "Table1: col1, col2\nTable2: col3, col4")
        mock_db.get_table_info.assert_called_once()
        
        # Test sample rows method
        result = agent.get_sample_rows("test_table", 5)
        self.assertEqual(result, "Sample data")
        mock_db.run.assert_called_once_with("SELECT * FROM test_table LIMIT 5")

if __name__ == "__main__":
    unittest.main() 