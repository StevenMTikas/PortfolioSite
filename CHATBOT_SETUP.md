# Chatbot Setup Instructions

This chatbot feature allows visitors to ask questions about Steven Tikas using information from documents in the `assets/MeBot` folder.

## Prerequisites

1. Python 3.8 or higher
2. OpenAI API key (get one from https://platform.openai.com/api-keys)

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root (or set environment variables):

```env
OPENAI_API_KEY=your_openai_api_key_here
PORT=8000  # Optional, defaults to 8000
```

### 3. Ensure Documents Are in Place

Make sure your PDF files are in the `assets/MeBot` folder:
- `assets/MeBot/Resume.pdf`
- `assets/MeBot/LinkedIn_Profile.pdf`

The server will automatically load all PDF files from this folder on startup.

### 4. Start the Backend Server

```bash
python chatbot_server.py
```

The server will start on `http://localhost:8000` by default.

You should see output like:
```
Loading documents...
Found 2 PDF file(s) to process...
Loading Resume.pdf...
  Loaded X pages from Resume.pdf
Loading LinkedIn_Profile.pdf...
  Loaded Y pages from LinkedIn_Profile.pdf
Total pages loaded: Z
Splitting documents into chunks...
Created N text chunks
Creating embeddings and vector store...
Vector store created successfully!
Initializing QA chain...
QA chain initialized successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5. Open Your Portfolio Site

Open `index.html` in your browser. The chatbot button should now be functional!

## How It Works

1. **Document Loading**: On startup, the server loads all PDF files from `assets/MeBot`
2. **Text Processing**: Documents are split into chunks for efficient retrieval
3. **Embeddings**: Text chunks are converted to embeddings using OpenAI's embedding model
4. **Vector Store**: Embeddings are stored in a FAISS vector database for fast similarity search
5. **Question Answering**: When a question is asked:
   - The system finds the most relevant document chunks
   - Uses those chunks as context for the LLM
   - Generates an answer based on the context

## API Endpoints

- `GET /` - Health check endpoint
- `POST /api/ask` - Ask a question
  - Request body: `{"message": "What is Steven's experience?"}`
  - Response: `{"reply": "Steven has..."}`

## Troubleshooting

### Server won't start
- Check that `OPENAI_API_KEY` is set correctly
- Ensure PDF files exist in `assets/MeBot` folder
- Check that all dependencies are installed

### Chatbot not responding
- Verify the backend server is running on port 8000
- Check browser console for errors
- Ensure CORS is properly configured (currently allows all origins)

### Slow responses
- The first request may be slower as the vector store initializes
- Consider using a faster model or reducing chunk size if needed

## Production Deployment

For production:
1. Change CORS settings in `chatbot_server.py` to allow only your domain
2. Use environment variables for sensitive configuration
3. Consider using a production ASGI server like Gunicorn with Uvicorn workers
4. Set up proper error logging and monitoring

