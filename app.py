import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from sql_agent import SQLChatAgent
from custom_llm import AnythingLLMProvider
from langchain_core.messages import HumanMessage
import pandas as pd
import re
import matplotlib.pyplot as plt
import io

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Chat with Database", layout="wide")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_connected" not in st.session_state:
    st.session_state.db_connected = False
if "llm_configured" not in st.session_state:
    st.session_state.llm_configured = False
if "llm" not in st.session_state:
    st.session_state.llm = None
if "tables" not in st.session_state:
    st.session_state.tables = []

# Function to extract table data from query results
def extract_table_data(text):
    """Extract tabular data from query results and convert to DataFrame."""
    # Look for data that appears to be in a table format with columns and rows
    lines = text.strip().split('\n')
    
    # Check if we have at least two lines for header and data
    if len(lines) < 2:
        return None
    
    # First check for Python list/tuple format like: [('Vitamins', 1), ('wisdom tooth', 1), ...]
    # This is common in SQL query results
    list_pattern = re.compile(r'\[\s*\(.*\).*\]')
    for line in lines:
        if list_pattern.search(line):
            try:
                # Extract the list part
                list_text = re.search(r'\[(.*)\]', line).group(1).strip()
                if not list_text:
                    continue
                
                # Try to safely evaluate the list of tuples
                import ast
                items = ast.literal_eval('[' + list_text + ']')
                
                if items and all(isinstance(item, tuple) for item in items) and len(items[0]) >= 2:
                    # Get column names from context
                    col_names = []
                    # Look for a SELECT statement before the results
                    for i in range(len(lines)):
                        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', '\n'.join(lines[:i+1]), re.IGNORECASE | re.DOTALL)
                        if select_match:
                            # Extract column names from the SELECT statement
                            columns_text = select_match.group(1)
                            col_candidates = [c.strip().split(' AS ')[-1].strip() for c in columns_text.split(',')]
                            # Take the last word of each column as the name
                            col_names = [c.split('.')[-1].strip() for c in col_candidates]
                            if len(col_names) == len(items[0]):
                                break
                    
                    # If we couldn't find column names, use generic ones
                    if not col_names or len(col_names) != len(items[0]):
                        col_names = [f"column_{i}" for i in range(len(items[0]))]
                    
                    # Create DataFrame and convert to appropriate types
                    df = pd.DataFrame(items, columns=col_names)
                    
                    # Try to convert columns to numeric where appropriate
                    for col in df.columns:
                        # Check if column has numeric values
                        if df[col].apply(lambda x: isinstance(x, (int, float))).all():
                            df[col] = pd.to_numeric(df[col])
                    
                    return df
            except Exception as e:
                print(f"Error parsing list/tuple format: {e}")
    
    # Check for Markdown tables first (they start with | and have a separator line with dashes)
    markdown_table_pattern = re.compile(r'^\s*\|(.+)\|\s*$')
    separator_pattern = re.compile(r'^\s*\|\s*[-:]+\s*\|\s*[-:]+')
    
    md_table_lines = []
    in_md_table = False
    header_found = False
    
    for i, line in enumerate(lines):
        if markdown_table_pattern.match(line):
            if not in_md_table:
                in_md_table = True
                md_table_lines.append(line)
                # Check if next line is a separator
                if i+1 < len(lines) and separator_pattern.match(lines[i+1]):
                    header_found = True
            else:
                md_table_lines.append(line)
        elif in_md_table and not line.strip():  # Empty line ends the table
            break
    
    if in_md_table and len(md_table_lines) >= 2:
        try:
            # Process Markdown table
            clean_lines = []
            for i, line in enumerate(md_table_lines):
                # Skip separator lines
                if i == 1 and header_found and separator_pattern.match(line):
                    continue
                # Clean up the line
                cleaned = re.sub(r'^\s*\|\s*|\s*\|\s*$', '', line)
                parts = [p.strip() for p in cleaned.split('|')]
                clean_lines.append(parts)
            
            if len(clean_lines) >= 2:
                df = pd.DataFrame(clean_lines[1:], columns=clean_lines[0])
                # Try to convert numeric columns
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except (ValueError, TypeError):
                        pass
                return df
        except Exception:
            pass
    
    # Try to detect SQL result tables with various formats
    # Look for common patterns like "| column1 | column2 |" or "+-------+-------+"
    table_start_patterns = [
        re.compile(r'^\s*\+[-+]+\+\s*$'),  # +----+----+
        re.compile(r'^\s*\|.*\|\s*$'),     # | data | data |
        re.compile(r'^\s*[A-Za-z0-9_]+\s+[A-Za-z0-9_]+')  # column1 column2 (space-separated)
    ]
    
    # Try to identify where the table starts
    table_start_idx = -1
    for i, line in enumerate(lines):
        for pattern in table_start_patterns:
            if pattern.match(line):
                table_start_idx = i
                break
        if table_start_idx >= 0:
            break
    
    if table_start_idx >= 0:
        # Extract the table portion
        table_lines = []
        i = table_start_idx
        while i < len(lines) and lines[i].strip():
            table_lines.append(lines[i])
            i += 1
            # Stop if we hit another markdown code block or non-table text
            if i < len(lines) and (lines[i].startswith('```') or 
                                   lines[i].startswith('#') or 
                                   (not lines[i].strip() and i+1 < len(lines) and not lines[i+1].strip())):
                break
                
        # Try to parse as various table formats
        try:
            # Handle tables with pipe separators
            if any('|' in line for line in table_lines[:3]):
                cleaned_lines = []
                
                for line in table_lines:
                    # Skip separator lines like +----+----+ or ---+---
                    if re.match(r'^[\s\+\-\|]+$', line):
                        continue
                    # Clean up the line
                    cleaned_line = re.sub(r'^\s*\|\s*|\s*\|\s*$', '', line)
                    cells = [cell.strip() for cell in cleaned_line.split('|')]
                    if cells and any(cell for cell in cells):  # Skip empty rows
                        cleaned_lines.append(cells)
                
                if len(cleaned_lines) >= 2:  # Header and at least one row
                    df = pd.DataFrame(cleaned_lines[1:], columns=cleaned_lines[0])
                    # Try to convert numeric columns
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except (ValueError, TypeError):
                            pass
                    return df
            
            # Handle space-aligned columns
            elif all(len(line.split()) >= 2 for line in table_lines[:3]):
                # Try to determine column positions by looking at consistent spaces
                # This is a simplified approach and might not work for all space-delimited tables
                header = table_lines[0]
                col_positions = [0]  # Start of first column
                
                # Find spaces in the header that might indicate column separators
                for i in range(1, len(header) - 1):
                    if header[i].isspace() and not header[i-1].isspace() and not header[i+1].isspace():
                        col_positions.append(i+1)
                
                if len(col_positions) >= 2:  # At least 2 columns
                    rows = []
                    for line in table_lines:
                        row = []
                        for i in range(len(col_positions)):
                            start = col_positions[i]
                            end = col_positions[i+1] if i < len(col_positions) - 1 else len(line)
                            row.append(line[start:end].strip())
                        rows.append(row)
                    
                    if len(rows) >= 2:  # Header and at least one row
                        df = pd.DataFrame(rows[1:], columns=rows[0])
                        # Try to convert numeric columns
                        for col in df.columns:
                            try:
                                df[col] = pd.to_numeric(df[col])
                            except (ValueError, TypeError):
                                pass
                        return df
        except Exception:
            pass
    
    # If we couldn't parse it as a structured table, try each delimiter as a last resort
    for delimiter in ['\t', ',', ';', '|']:
        try:
            if any(delimiter in line for line in lines[:5]):
                # Get the header and data lines
                header_line = next((line for line in lines if delimiter in line), None)
                if not header_line:
                    continue
                    
                header_idx = lines.index(header_line)
                data_lines = [line for line in lines[header_idx:] if delimiter in line]
                
                if len(data_lines) >= 2:  # Header and at least one row
                    df = pd.DataFrame([line.split(delimiter) for line in data_lines])
                    df.columns = df.iloc[0].str.strip()
                    df = df[1:].reset_index(drop=True)
                    # Try to convert numeric columns
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except (ValueError, TypeError):
                            pass
                    return df
        except Exception:
            continue
    
    # Last resort: Try to parse any text that looks like CSV or TSV data
    try:
        # Look for a section that might be CSV/TSV data
        csv_lines = []
        for i, line in enumerate(lines):
            # Check if line has commas or tabs that might indicate CSV/TSV
            if ',' in line or '\t' in line:
                csv_lines.append(line)
                
        if len(csv_lines) >= 2:
            # Try to determine delimiter (comma or tab)
            delimiter = '\t' if '\t' in csv_lines[0] else ','
            df = pd.read_csv(io.StringIO('\n'.join(csv_lines)), sep=delimiter)
            return df
    except Exception:
        pass
    
    return None

# Add visualization functions
def render_visualizations(df, unique_id="default"):
    """Render appropriate visualizations based on the dataframe."""
    if df is None or df.empty or len(df) < 2:
        return
    
    st.subheader("Data Visualization")
    
    # Check if we have numeric columns for visualization
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    if not numeric_cols:
        st.write("No numeric columns available for visualization")
        # Try to convert a column if possible
        if len(df.columns) >= 2:
            try:
                count_col = df.columns[1]  # Often the second column is a count or numeric value
                df[count_col] = pd.to_numeric(df[count_col], errors='coerce')
                if not df[count_col].isna().all():
                    numeric_cols = [count_col]
                    st.success(f"Successfully converted '{count_col}' column to numeric for visualization")
                else:
                    return
            except:
                return
        else:
            return
    
    # Display the raw data as a table first for reference
    st.subheader("Data Table")
    st.dataframe(df)
    
    # Let user select visualization type
    st.subheader("Select Visualization")
    
    # Determine which visualizations make sense for this data
    available_charts = ["Bar Chart"]
    if len(df) > 1:
        available_charts.extend(["Pie Chart"])
    if len(df) > 2:
        available_charts.extend(["Line Chart", "Area Chart"])
    if len(numeric_cols) > 1:
        available_charts.append("Scatter Plot")
    available_charts.append("Histogram")
    
    viz_type = st.selectbox(
        "Chart Type",
        available_charts,
        key=f"viz_type_{unique_id}"
    )
    
    # Special handling for count/frequency data which is common in SQL queries
    count_viz_mode = False
    if len(categorical_cols) == 1 and len(numeric_cols) == 1:
        # This looks like a frequency distribution or count query
        count_viz_mode = True
        label_col = categorical_cols[0]
        value_col = numeric_cols[0]
    
    # Configuration options based on chart type
    if viz_type == "Bar Chart":
        if count_viz_mode:
            # For count data, use the categorical column as x-axis by default
            st.bar_chart(df, x=label_col, y=value_col)
        else:
            x_axis = st.selectbox("X-axis", df.columns.tolist(), key=f"x_axis_{unique_id}")
            y_axis = st.multiselect("Y-axis", numeric_cols, default=[numeric_cols[0]] if numeric_cols else [], key=f"y_axis_{unique_id}")
            
            if not y_axis:
                st.warning("Please select at least one Y-axis column")
                return
            
            st.bar_chart(df, x=x_axis, y=y_axis)
    
    elif viz_type in ["Line Chart", "Area Chart"]:
        if count_viz_mode:
            # For count data, use the categorical column as x-axis by default
            if viz_type == "Line Chart":
                st.line_chart(df, x=label_col, y=value_col)
            else:
                st.area_chart(df, x=label_col, y=value_col)
        else:
            x_axis = st.selectbox("X-axis", df.columns.tolist(), key=f"x_axis_{unique_id}")
            y_axis = st.multiselect("Y-axis", numeric_cols, default=[numeric_cols[0]] if numeric_cols else [], key=f"y_axis_{unique_id}")
            
            if not y_axis:
                st.warning("Please select at least one Y-axis column")
                return
            
            if viz_type == "Line Chart":
                st.line_chart(df, x=x_axis, y=y_axis)
            else:
                st.area_chart(df, x=x_axis, y=y_axis)
    
    elif viz_type == "Scatter Plot":
        x_col = st.selectbox("X-axis", numeric_cols, key=f"scatter_x_{unique_id}")
        y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key=f"scatter_y_{unique_id}")
        st.scatter_chart(df, x=x_col, y=y_col)
    
    elif viz_type == "Pie Chart":
        if count_viz_mode:
            # For count data, use categorical column for labels and numeric for values by default
            fig, ax = plt.subplots(figsize=(10, 6))
            explode = [0.05] * len(df)  # Small explode for each slice
            wedges, texts, autotexts = ax.pie(
                df[value_col], 
                labels=df[label_col], 
                autopct='%1.1f%%',
                explode=explode,
                shadow=True,
                startangle=90
            )
            # Make the labels more readable
            plt.setp(autotexts, size=10, weight="bold")
            plt.setp(texts, size=9)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.tight_layout()
            st.pyplot(fig)
        else:
            if len(categorical_cols) > 0:
                label_col = st.selectbox("Label Column", categorical_cols, key=f"pie_label_{unique_id}")
                value_col = st.selectbox("Value Column", numeric_cols, key=f"pie_value_{unique_id}")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                df_grouped = df.groupby(label_col).sum().reset_index()
                explode = [0.05] * len(df_grouped)  # Small explode for each slice
                wedges, texts, autotexts = ax.pie(
                    df_grouped[value_col], 
                    labels=df_grouped[label_col], 
                    autopct='%1.1f%%',
                    explode=explode,
                    shadow=True,
                    startangle=90
                )
                # Make the labels more readable
                plt.setp(autotexts, size=10, weight="bold")
                plt.setp(texts, size=9)
                ax.axis('equal')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Need at least one categorical column for pie chart labels")
    
    elif viz_type == "Histogram":
        hist_column = st.selectbox("Column", numeric_cols, key=f"hist_column_{unique_id}")
        bins = st.slider("Number of bins", min_value=5, max_value=min(100, len(df)), value=min(20, len(df)), key=f"hist_bins_{unique_id}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df[hist_column].hist(bins=bins, ax=ax)
        ax.set_title(f'Histogram of {hist_column}')
        ax.set_xlabel(hist_column)
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Database Configuration Section
    st.subheader("Database Setup")
    db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQLite"], index=0)
    
    if db_type == "PostgreSQL":
        db_host = st.text_input("Host", value="localhost")
        db_port = st.text_input("Port", value="5432")
        db_name = st.text_input("Database Name")
        db_user = st.text_input("Username")
        db_password = st.text_input("Password", type="password")
        
        db_connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == "MySQL":
        db_host = st.text_input("Host", value="localhost")
        db_port = st.text_input("Port", value="3306")
        db_name = st.text_input("Database Name")
        db_user = st.text_input("Username")
        db_password = st.text_input("Password", type="password")
        
        db_connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == "SQLite":
        db_path = st.text_input("Database File Path", value="sqlite.db")
        db_connection_string = f"sqlite:///{db_path}"
    
    # Comment out env var check to use UI inputs
    # if os.getenv("DB_CONNECTION_STRING"):
    #     db_connection_string = os.getenv("DB_CONNECTION_STRING")
    #     st.info("Using database connection from .env file")
    
    test_db_connection = st.button("Test Database Connection")
    
    if test_db_connection:
        try:
            engine = create_engine(db_connection_string)
            connection = engine.connect()
            connection.close()
            st.session_state.db_connected = True
            st.session_state.db_connection_string = db_connection_string
            
            # Get table information
            if "llm" in st.session_state:
                try:
                    sql_agent = SQLChatAgent(st.session_state.llm, db_connection_string)
                    table_info = sql_agent.get_table_info()
                    st.session_state.tables = table_info
                except Exception as e:
                    st.error(f"Could not fetch table information: {e}")
            
            st.success("Database connection successful!")
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            st.session_state.db_connected = False
    
    # LLM Configuration Section
    st.subheader("LLM Setup")
    
    llm_provider = st.selectbox("LLM Provider", ["OpenAI", "Azure OpenAI", "AnythingLLM"])
    
    if llm_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model_name = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"], index=0)
    elif llm_provider == "Azure OpenAI":
        api_key = st.text_input("Azure OpenAI API Key", type="password", 
                                value=os.getenv("AZURE_OPENAI_API_KEY", ""))
        azure_endpoint = st.text_input("Azure Endpoint", 
                                       value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
        deployment_name = st.text_input("Deployment Name", 
                                        value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""))
        model_name = deployment_name  # For Azure, the deployment name is used
    elif llm_provider == "AnythingLLM":
        api_url = st.text_input("API URL", 
                                value=os.getenv("ANYTHING_LLM_URL", "https://5711-2409-40c2-1229-7cf5-2cdf-a14c-72c4-cccf.ngrok-free.app/api/v1/openai/chat/completions"))
        api_key = st.text_input("API Key", type="password", 
                                value=os.getenv("ANYTHING_LLM_API_KEY", "A2E1TVH-TMYMJXN-HBTS93Z-5J5ZZ2W"))
        model_name = st.text_input("Model Name", value=os.getenv("ANYTHING_LLM_MODEL", "demo"))
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Advanced configuration for AnythingLLM
        with st.expander("Advanced configuration"):
            system_prompt = st.text_area(
                "System Prompt", 
                value=os.getenv("ANYTHING_LLM_SYSTEM_PROMPT", 
                                "You are an expert SQL agent that helps users analyze their database by translating their natural language questions into SQL queries. "
                                "When responding to user queries:\n"
                                "1. Always include the SQL query you generated to answer the question\n"
                                "2. Present results in a clear, human-readable format\n"
                                "3. If relevant, provide brief explanations of the results\n"
                                "4. If there are ambiguities in the user's question, ask for clarification\n"
                                "5. If the question cannot be answered with the available schema, explain why\n\n"
                                "Remember to consider the database schema and use appropriate SQL syntax for the specific database type.")
            )
    
    test_llm_config = st.button("Test LLM Configuration")
    
    if test_llm_config:
        try:
            if llm_provider == "OpenAI":
                llm = ChatOpenAI(api_key=api_key, model_name=model_name)
                st.session_state.llm = llm
                st.session_state.llm_configured = True
                st.success("LLM configuration successful!")
            elif llm_provider == "Azure OpenAI":
                # Set the environment variables for Azure OpenAI
                os.environ["AZURE_OPENAI_API_KEY"] = api_key
                os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
                
                llm = ChatOpenAI(
                    deployment_name=deployment_name,
                    model_name=model_name,
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version="2023-05-15"
                )
                st.session_state.llm = llm
                st.session_state.llm_configured = True
                st.success("LLM configuration successful!")
            elif llm_provider == "AnythingLLM":
                # Now AnythingLLMProvider is a BaseChatModel, not an LLM
                llm = AnythingLLMProvider(
                    api_url=api_url,
                    api_key=api_key,
                    model=model_name,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                # Test the chat model with a simple message
                try:
                    test_message = [HumanMessage(content="What is 2+2?")]
                    test_response = llm.invoke(test_message)
                    st.session_state.llm = llm
                    st.session_state.llm_configured = True
                    st.success(f"AnythingLLM configuration successful! Test response: {test_response.content[:100]}...")
                except Exception as e:
                    st.error(f"AnythingLLM test failed: {e}")
                    st.session_state.llm_configured = False
                
            # If database is already connected, try to get table info
            if st.session_state.db_connected:
                try:
                    sql_agent = SQLChatAgent(st.session_state.llm, st.session_state.db_connection_string)
                    table_info = sql_agent.get_table_info()
                    st.session_state.tables = table_info
                except Exception as e:
                    st.error(f"Could not fetch table information: {e}")
                    
        except Exception as e:
            st.error(f"LLM configuration failed: {e}")
            st.session_state.llm_configured = False
    
    st.markdown("---")
    
    if st.session_state.db_connected and st.session_state.llm_configured:
        status = st.success("Ready to chat with your database!")
    else:
        if not st.session_state.db_connected:
            st.warning("Database not connected")
        if not st.session_state.llm_configured:
            st.warning("LLM not configured")

# Main app
st.title("Chat with Your Database")

# Create tabs for Chat and Schema
tab1, tab2 = st.tabs(["Chat", "Database Schema"])

with tab1:
    # Display conversation history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Check if this is an assistant message
            if message["role"] == "assistant":
                # Always show the visualization expander
                with st.expander("View Data Visualizations"):
                    # Try to extract table data from the response
                    df = extract_table_data(message["content"])
                    if df is not None and not df.empty:
                        render_visualizations(df, unique_id=f"history_{i}")
                    else:
                        st.info("No tabular data detected in this response. You can try another query that returns a data table.")
                        # Offer a text area to paste data manually
                        manual_data = st.text_area(
                            "Or paste tabular data here to visualize (CSV, TSV, or pipe-delimited):", 
                            height=150, 
                            key=f"history_manual_data_{i}"  # Unique key
                        )
                        if manual_data.strip():
                            df_manual = extract_table_data(manual_data)
                            if df_manual is not None and not df_manual.empty:
                                render_visualizations(df_manual, unique_id=f"history_manual_data_{i}")
                            else:
                                st.error("Could not parse the provided data as a table. Please ensure it's in a tabular format.")

    # Input field for user's question
    if st.session_state.db_connected and st.session_state.llm_configured:
        user_question = st.chat_input("Ask a question about your database")
        
        if user_question:
            # Add user's question to conversation history
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Display user's question
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Display assistant's response (with a spinner while generating)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        # Get response from SQL agent
                        sql_agent = SQLChatAgent(st.session_state.llm, st.session_state.db_connection_string)
                        response_content = sql_agent.query(user_question)
                        
                        # Display response
                        message_placeholder.markdown(response_content)
                        
                        # Always show the visualization expander
                        with st.expander("View Data Visualizations"):
                            # Try to extract table data from the response
                            df = extract_table_data(response_content)
                            if df is not None and not df.empty:
                                render_visualizations(df, unique_id=f"current_response")
                            else:
                                st.info("No tabular data detected in this response. You can try another query that returns a data table.")
                                # Offer a text area to paste data manually
                                manual_data = st.text_area(
                                    "Or paste tabular data here to visualize (CSV, TSV, or pipe-delimited):", 
                                    height=150,
                                    key="current_manual_data"  # Unique key
                                )
                                if manual_data.strip():
                                    df_manual = extract_table_data(manual_data)
                                    if df_manual is not None and not df_manual.empty:
                                        render_visualizations(df_manual, unique_id="current_manual_data")
                                    else:
                                        st.error("Could not parse the provided data as a table. Please ensure it's in a tabular format.")
                        
                        # Add response to conversation history
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    except Exception as e:
                        error_message = f"Error processing your query: {str(e)}"
                        message_placeholder.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        st.info("Please configure the database and LLM in the sidebar to start chatting.")

with tab2:
    # Display schema information
    st.subheader("Database Schema")
    if st.session_state.db_connected:
        if "tables" in st.session_state and st.session_state.tables:
            st.code(st.session_state.tables)
        else:
            try:
                if "llm" in st.session_state and st.session_state.llm is not None:
                    sql_agent = SQLChatAgent(st.session_state.llm, st.session_state.db_connection_string)
                    table_info = sql_agent.get_table_info()
                    st.session_state.tables = table_info
                    st.code(table_info)
                else:
                    st.warning("LLM must be configured to fetch schema information")
            except Exception as e:
                st.error(f"Could not fetch schema information: {e}")
    else:
        st.info("Connect to a database to view schema information") 