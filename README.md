# DB Explorer

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
- File analysis with Google Gemini AI:
  - Upload and analyze PDF documents
  - Upload and analyze images
  - Extract data for database queries
  - Use extracted IDs and information as context for database searches

## Prerequisites

- Python 3.8+
- A database (PostgreSQL, MySQL, or SQLite)
- One of the following:
  - OpenAI API key
  - Azure OpenAI credentials
  - AnythingLLM API endpoint and key
- Google Gemini API key (for file analysis features)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/gastrosurgeryailabs/db-insight-explorer.git
   cd db-insight-explorer
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

5. To analyze files:
   - Configure your Gemini API key in the sidebar
   - Upload a PDF or image using the file upload buttons
   - Click "Analyze File" to process with Gemini AI
   - Choose "Use for Database Query" to use the extracted data in your queries
   - Ask questions that reference the extracted information

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
- **Google Gemini AI**: For PDF and image analysis

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

## File Analysis with Gemini AI

The application provides advanced file analysis capabilities powered by Google's Gemini AI:

### PDF Analysis
- Upload PDF documents to extract IDs, reference numbers, and key information
- Extracted data is automatically used as context for database queries
- Specialized prompts for different document types (invoices, ID cards, etc.)

### Image Analysis
- Upload images to extract IDs, product codes, and other identifiers
- Perfect for game screenshots, product images, or ID cards
- Extracted information becomes context for your database queries

### Integration with Database Queries
- Upload a file containing IDs or reference numbers
- Ask questions about the extracted information
- The application combines the extracted data with your question
- Results show the relevant database information for the extracted IDs

### Configuration
- Set your Gemini API key in the sidebar
- Choose from specialized prompts for different document types
- Customize analysis prompts for specific use cases

## Examples of Questions You Can Ask

- "How many users are registered in the system?"
- "What are the top 5 products by sales in the last month?"
- "Show me the average order value grouped by customer segment"
- "Which customers made more than 10 purchases?"

## Troubleshooting

- **Database Connection Issues**: Make sure your database is running and accessible from your network
- **LLM Configuration Errors**: Verify your API keys and endpoints are correct
- **Query Errors**: If you get errors on specific queries, try rephrasing your question or providing more context
- **File Analysis Issues**: 
  - Ensure your Gemini API key is valid
  - Check that file formats are supported (PDF, JPG, JPEG, PNG)
  - Large files may take longer to process