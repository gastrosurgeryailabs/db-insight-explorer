# Chat with Database

A Streamlit application that allows you to chat with your database using natural language. Ask questions about your data and get answers through an AI-powered interface with interactive visualizations.

## Features

- Connect to PostgreSQL, MySQL, or SQLite databases
- Configure OpenAI, Azure OpenAI, or AnythingLLM as your LLM provider
- Natural language interface to query your database
- Interactive data visualizations including:
  - Bar charts
  - Line charts
  - Area charts
  - Scatter plots
  - Pie charts
  - Histograms
- Smart visualization suggestions based on data type
- Chat history to review previous questions and answers
- Easy-to-use configuration in the sidebar

## Prerequisites

- Python 3.8+
- A database (PostgreSQL, MySQL, or SQLite)
- One of the following:
  - OpenAI API key
  - Azure OpenAI credentials
  - AnythingLLM API endpoint and key

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/gastrosurgeryailabs/db-insight-explorer.git
   cd chat-with-db
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file by copying the example:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file with your credentials.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser at `http://localhost:8501`

3. Configure your database and LLM settings in the sidebar

4. Start chatting with your database!

## How it Works

This application uses:
- **Streamlit**: For the web interface and interactive visualizations
- **LangChain**: For creating the SQL agent
- **SQLAlchemy**: For database connections
- **Matplotlib**: For custom data visualizations
- **LLM Providers**:
  - OpenAI
  - Azure OpenAI
  - AnythingLLM (with custom API endpoint support)

The app translates your natural language questions into SQL queries and returns the results in a conversational format with appropriate data visualizations.

## Data Visualization

The application automatically suggests and generates appropriate visualizations based on your query results:

- **Numeric Data**: Automatically detected and visualized using appropriate chart types
- **Interactive Controls**: Customize your visualizations with:
  - Chart type selection
  - Axis configuration
  - Binning options for histograms
  - Multiple series support
- **Smart Defaults**: Automatic selection of suitable visualization types based on data structure
- **Export Options**: View and interact with visualizations directly in the interface

## LLM Provider Options

### OpenAI
Standard OpenAI API integration with support for different models (GPT-4, GPT-3.5, etc.)

### Azure OpenAI
For users using Azure OpenAI deployments, with custom endpoint and deployment name support.

### AnythingLLM
Custom LLM provider option that works with your own AnythingLLM API endpoint. This allows you to use any LLM that's compatible with the OpenAI API format.

## Examples of Questions You Can Ask

- "How many users are registered in the system?"
- "What are the top 5 products by sales in the last month?"
- "Show me the average order value grouped by customer segment"
- "Which customers made more than 10 purchases?"

## Troubleshooting

- **Database Connection Issues**: Make sure your database is running and accessible from your network
- **LLM Configuration Errors**: Verify your API keys and endpoints are correct
- **Query Errors**: If you get errors on specific queries, try rephrasing your question or providing more context