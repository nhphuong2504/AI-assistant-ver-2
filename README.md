# AI analyst - Retail Data Chat Application

A full-stack monorepo application combining a FastAPI backend with a React frontend for AI-powered retail data analysis and chat interactions.

## 📁 Project Structure

```
assistant/
├── backend/              # FastAPI backend application
│   ├── app/             # Main application code
│   │   ├── main.py      # FastAPI routes and endpoints
│   │   ├── db.py        # Database utilities
│   │   └── llm_langchain.py  # LangChain AI agent
│   ├── analytics/       # Analytics modules (CLV, survival analysis)
│   ├── etl/             # Data loading scripts
│   ├── data/            # Data files (CSV, SQLite)
│   └── test/            # Backend tests
├── frontend/            # React + TypeScript frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   └── lib/         # API utilities
│   └── public/          # Static assets
├── venv/                # Python virtual environment (root level)
├── requirements.txt     # Python dependencies
├── setup-backend.bat    # Backend setup script
├── start-backend.bat    # Backend startup script
└── start-frontend.bat   # Frontend startup script
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed
- **Node.js 18+** and npm installed
- **OpenAI API Key** (for AI features)

### Backend Setup

1. **Run the setup script:**
   ```bash
   setup-backend.bat
   ```
   This will:
   - Create a Python virtual environment at the root (`venv/`)
   - Install all Python dependencies from `requirements.txt`

2. **Configure environment variables:**
   - Copy `backend/.env.example` to `backend/.env`
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     OPENAI_MODEL=gpt-4
     API_URL=http://127.0.0.1:8000
     DATABASE_PATH=backend/data/retail.sqlite
     ```

3. **Generate the database:**
   The `retail.sqlite` database file is not included in the repository. You need to generate it from the CSV data:
   
   **Windows:**
   ```bash
   cd backend
   ..\venv\Scripts\python etl\load_online_retail.py
   ```
   
   **Linux/Mac:**
   ```bash
   cd backend
   ../venv/bin/python etl/load_online_retail.py
   ```
   
   This script will:
   - Read the CSV file from `backend/data/raw/online_retail.csv`
   - Clean and process the data
   - Create `backend/data/retail.sqlite` with two tables:
     - `transactions_all` - All raw transactions with data quality flags
     - `transactions` - Clean transactions for analytics (excludes cancellations, returns, invalid data)
   - Create indexes for optimal query performance

4. **Start the backend server:**
   ```bash
   start-backend.bat
   ```
   The backend will run on `http://127.0.0.1:8000`

### Frontend Setup

1. **Start the frontend development server:**
   ```bash
   start-frontend.bat
   ```
   This will automatically install Node.js dependencies if needed and start the dev server.

2. **Configure environment variables (optional):**
   - Copy `frontend/.env.example` to `frontend/.env`
   - Set the API URL (defaults to `http://127.0.0.1:8000`):
     ```
     VITE_API_URL=http://127.0.0.1:8000
     ```

The frontend will run on `http://localhost:8080`

## 🐍 Virtual Environment

**IMPORTANT:** The Python virtual environment (`venv/`) is located at the **root level** of the project, not in the `backend/` directory.

### Manual Activation (if needed)

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

After activation, you can run Python commands directly. The setup and startup scripts handle activation automatically.

## 🔧 Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LangChain** - AI agent framework
- **LangGraph** - Agent orchestration
- **OpenAI** - LLM integration
- **SQLite** - Database
- **Pandas** - Data manipulation
- **Uvicorn** - ASGI server

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **React Router** - Routing
- **TanStack Query** - Data fetching

## 📡 API Endpoints

### Main Endpoints

- `POST /ask-langchain` - Ask questions to the AI assistant
  ```json
  {
    "question": "What are the top customers?",
    "use_memory": true,
    "thread_id": "default"
  }
  ```

- `POST /query` - Execute SQL queries (read-only)
- `GET /schema` - Get database schema
- `POST /clear-memory` - Clear conversation memory

See `backend/app/main.py` for the complete API documentation.

## 🛠️ Development

### Backend Development

The backend uses FastAPI with auto-reload enabled. Changes to Python files will automatically restart the server.

### Database Setup

If you need to regenerate the `retail.sqlite` database (e.g., after modifying the ETL script or CSV data):

**Windows:**
```bash
cd backend
..\venv\Scripts\python etl\load_online_retail.py
```

**Linux/Mac:**
```bash
cd backend
../venv/bin/python etl/load_online_retail.py
```

**Note:** The database file (`backend/data/retail.sqlite`) is excluded from version control via `.gitignore`. Each developer needs to generate it locally using the ETL script.

### Frontend Development

The frontend uses Vite with hot module replacement. Changes to React components will update in the browser automatically.

### Running Tests

**Backend:**
```bash
cd backend
pytest
```

**Frontend:**
```bash
cd frontend
npm test
```

## 📝 Configuration

### Backend Environment Variables

Create `backend/.env` with:
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_MODEL` - Model to use (default: `gpt-4`)
- `API_URL` - Backend URL (default: `http://127.0.0.1:8000`)
- `DATABASE_PATH` - Path to SQLite database

### Frontend Environment Variables

Create `frontend/.env` with:
- `VITE_API_URL` - Backend API URL (default: `http://127.0.0.1:8000`)

## 🎯 Features

- **AI-Powered Chat Interface** - Natural language queries about retail data
- **Conversation Memory** - Context-aware responses using thread IDs
- **SQL Query Interface** - Direct database querying (read-only)
- **Analytics Integration** - CLV and survival analysis
- **CORS Enabled** - Frontend-backend communication configured

## 📚 Additional Resources

- FastAPI Documentation: https://fastapi.tiangolo.com/
- LangChain Documentation: https://python.langchain.com/
- React Documentation: https://react.dev/
- Vite Documentation: https://vitejs.dev/

## 🤝 Contributing

1. Ensure the virtual environment is activated
2. Make changes in the respective `backend/` or `frontend/` directories
3. Test your changes locally
4. Commit and push

---

**Note:** Always ensure the root-level `venv/` is used. Do not create virtual environments in subdirectories.
