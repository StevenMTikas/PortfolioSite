"""
FastAPI server for chatbot that answers questions about Steven Tikas
using documents from assets/MeBot folder.
"""
import os
import sys
import base64
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
env_path = script_dir / ".env"
load_dotenv(dotenv_path=env_path)

# Also try loading from current directory as fallback
load_dotenv()

# LangChain imports - using correct paths for LangChain 1.0+
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
# No prompt template imports needed - using direct string formatting

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    print("=" * 50)
    print("Starting chatbot server initialization...")
    print("=" * 50)
    try:
        initialize_vector_store()
        print("=" * 50)
        print("✓ Server initialization complete!")
        print(f"✓ Vector store: {'Loaded' if vector_store is not None else 'Not loaded'}")
        print(f"✓ LLM: {'Initialized' if llm is not None else 'Not initialized'}")
        print(f"✓ Retriever: {'Ready' if retriever is not None else 'Not ready'}")
        print("=" * 50)
    except Exception as e:
        print("=" * 50)
        print(f"✗ ERROR: Could not initialize vector store: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        print("The server will start, but chat functionality may not work.")
        print("Please check the error above and fix the issue.")
        print("=" * 50)
    
    yield
    
    # Shutdown (if needed)
    print("Shutting down chatbot server...")

app = FastAPI(title="Steven Tikas Chatbot API", lifespan=lifespan)

# CORS middleware - allow all origins for now (update for production)
# In production, replace ["*"] with your specific domains
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from chatbot directory
try:
    chatbot_dir = Path(__file__).parent / "chatbot"
    if chatbot_dir.exists():
        app.mount("/static", StaticFiles(directory=str(chatbot_dir)), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

# Root route - serve the chatbot HTML page
@app.get("/")
async def read_root():
    """Serve the chatbot HTML page."""
    chatbot_html = Path(__file__).parent / "chatbot" / "index.html"
    if chatbot_html.exists():
        return FileResponse(str(chatbot_html))
    else:
        return {"message": "Chatbot API is running", "status": "ok"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Chatbot API is running",
        "vector_store_loaded": vector_store is not None,
        "llm_initialized": llm is not None,
        "retriever_ready": retriever is not None,
        "ready": vector_store is not None and llm is not None and retriever is not None
    }

# Global variables for vector store and LLM
vector_store: Optional[FAISS] = None
llm: Optional[ChatOpenAI] = None
retriever = None
embeddings: Optional[OpenAIEmbeddings] = None  # Store embeddings for reuse
github_documents_loaded = False
github_loading_in_progress = False

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

def load_documents() -> List:
    """Load all documents from assets/MeBot folder, prioritizing bio.txt."""
    from langchain_community.document_loaders import TextLoader
    
    documents = []
    mebot_folder = Path("assets/MeBot")
    
    if not mebot_folder.exists():
        raise FileNotFoundError(f"MeBot folder not found at {mebot_folder}")
    
    # First, load bio.txt as the primary source (highest priority)
    bio_file = mebot_folder / "bio.txt"
    if bio_file.exists():
        print(f"Loading {bio_file.name} (PRIMARY SOURCE)...")
        try:
            loader = TextLoader(str(bio_file), encoding='utf-8')
            docs = loader.load()
            # Add metadata to mark this as the primary source
            for doc in docs:
                doc.metadata['source'] = 'bio.txt'
                doc.metadata['priority'] = 'primary'
            documents.extend(docs)
            print(f"  ✓ Loaded {len(docs)} pages from {bio_file.name} (PRIMARY SOURCE)")
        except Exception as e:
            print(f"  ✗ Error loading {bio_file.name}: {e}")
    else:
        print(f"  ⚠ Warning: {bio_file.name} not found. Continuing with other documents...")
    
    # Then load all PDF files (secondary sources)
    pdf_files = list(mebot_folder.glob("*.pdf"))
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF file(s) to process...")
        for pdf_file in pdf_files:
            print(f"Loading {pdf_file.name}...")
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                # Add metadata to mark these as tertiary sources (GitHub is secondary)
                for doc in docs:
                    doc.metadata['priority'] = 'tertiary'
                documents.extend(docs)
                print(f"  ✓ Loaded {len(docs)} pages from {pdf_file.name}")
            except Exception as e:
                print(f"  ✗ Error loading {pdf_file.name}: {e}")
                continue
    else:
        print("  No PDF files found in MeBot folder")
    
    if not documents:
        raise FileNotFoundError(f"No documents loaded from {mebot_folder}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    print(f"  - Primary source (bio.txt): {sum(1 for d in documents if d.metadata.get('priority') == 'primary')}")
    print(f"  - Tertiary sources (PDFs): {sum(1 for d in documents if d.metadata.get('priority') == 'tertiary')}")
    
    return documents

def load_github_repositories(username: str) -> List[Dict[str, Any]]:
    """
    Fetch all public repositories for a GitHub user.
    
    Args:
        username: GitHub username
        
    Returns:
        List of repository dictionaries with name, description, and README content
    """
    repos = []
    github_token = os.getenv("GITHUB_TOKEN")
    
    # Prepare headers
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Chatbot-GitHub-Integration"
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    try:
        # Fetch repositories
        print(f"Fetching repositories from GitHub for user: {username}...")
        url = f"https://api.github.com/users/{username}/repos"
        params = {"type": "all", "sort": "updated", "per_page": 100}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        # Check rate limit
        if response.status_code == 403:
            rate_limit = response.headers.get("X-RateLimit-Remaining", "unknown")
            print(f"  ⚠ GitHub API rate limit: {rate_limit} requests remaining")
            if rate_limit == "0":
                print("  ⚠ Rate limit exceeded. Consider setting GITHUB_TOKEN for higher limits.")
                return repos
        
        if response.status_code != 200:
            print(f"  ✗ Error fetching repositories: {response.status_code} - {response.text[:200]}")
            return repos
        
        repositories = response.json()
        print(f"  ✓ Found {len(repositories)} repositories")
        
        # Fetch README for each repository
        for repo in repositories:
            repo_name = repo.get("name", "")
            repo_description = repo.get("description", "") or "No description"
            repo_url = repo.get("html_url", "")
            repo_full_name = repo.get("full_name", "")
            
            readme_content = ""
            try:
                # Try to fetch README
                readme_url = f"https://api.github.com/repos/{repo_full_name}/readme"
                readme_response = requests.get(readme_url, headers=headers, timeout=10)
                
                if readme_response.status_code == 200:
                    readme_data = readme_response.json()
                    if readme_data.get("encoding") == "base64":
                        readme_content = base64.b64decode(readme_data.get("content", "")).decode("utf-8")
                    else:
                        readme_content = readme_data.get("content", "")
                elif readme_response.status_code == 404:
                    readme_content = "No README file found"
                else:
                    readme_content = f"Could not fetch README (status: {readme_response.status_code})"
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                readme_content = f"Error fetching README: {str(e)}"
            
            repos.append({
                "name": repo_name,
                "description": repo_description,
                "url": repo_url,
                "readme": readme_content,
                "full_name": repo_full_name
            })
        
        print(f"  ✓ Successfully loaded {len(repos)} repositories with READMEs")
        return repos
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Network error fetching GitHub repositories: {e}")
        return repos
    except Exception as e:
        print(f"  ✗ Error loading GitHub repositories: {e}")
        return repos

def fetch_github_documents(username: str) -> List:
    """
    Fetch GitHub repositories and convert them to LangChain Document objects.
    
    Args:
        username: GitHub username
        
    Returns:
        List of LangChain Document objects with GitHub repository information
    """
    from langchain_core.documents import Document
    
    documents = []
    repos = load_github_repositories(username)
    
    if not repos:
        print("  ⚠ No GitHub repositories loaded")
        return documents
    
    print(f"Converting {len(repos)} repositories to documents...")
    
    for repo in repos:
        # Format content
        content_parts = [
            f"Repository: {repo['name']}",
            f"Description: {repo['description']}",
            f"URL: {repo['url']}",
        ]
        
        if repo['readme'] and repo['readme'] not in ["No README file found", ""]:
            # Limit README size to avoid too large documents
            readme_text = repo['readme'][:5000]  # Limit to 5000 chars
            if len(repo['readme']) > 5000:
                readme_text += "\n\n[README truncated...]"
            content_parts.append(f"\nREADME:\n{readme_text}")
        
        content = "\n".join(content_parts)
        
        # Create document
        doc = Document(
            page_content=content,
            metadata={
                "source": "github",
                "priority": "secondary",
                "repo_name": repo['name'],
                "repo_url": repo['url'],
                "repo_full_name": repo['full_name']
            }
        )
        documents.append(doc)
    
    print(f"  ✓ Created {len(documents)} GitHub documents")
    return documents

def add_documents_to_vector_store(documents: List) -> None:
    """
    Add new documents to the existing FAISS vector store.
    
    Args:
        documents: List of LangChain Document objects to add
    """
    global vector_store, retriever, embeddings
    
    if not documents:
        return
    
    if vector_store is None:
        print("  ⚠ Vector store not initialized. Cannot add documents.")
        return
    
    if embeddings is None:
        print("  ⚠ Embeddings not initialized. Cannot add documents.")
        return
    
    try:
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"  Created {len(chunks)} text chunks from new documents")
        
        # Add documents to existing vector store
        # The vector store already has embeddings, so we can add documents directly
        # Note: We need to ensure the embeddings are the same instance
        try:
            # Try add_documents first (newer LangChain versions)
            vector_store.add_documents(chunks)
        except TypeError:
            # Fallback: use add_texts with embeddings if add_documents doesn't work
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            vector_store.add_texts(texts, metadatas=metadatas, embedding=embeddings)
        print(f"  ✓ Added {len(chunks)} chunks to vector store")
        
        # Update retriever to use the updated vector store
        # Increase k to retrieve more documents (helps ensure GitHub docs are found)
        retriever = vector_store.as_retriever(search_kwargs={"k": 8})
        print(f"  ✓ Retriever updated (vector store now has more documents, retrieving up to 8 docs)")
        
    except Exception as e:
        print(f"  ✗ Error adding documents to vector store: {e}")
        import traceback
        traceback.print_exc()

def ensure_github_loaded() -> None:
    """
    Ensure GitHub repositories are loaded into the vector store.
    Loads on-demand if not already loaded.
    """
    global github_documents_loaded, github_loading_in_progress
    
    if github_documents_loaded:
        return
    
    if github_loading_in_progress:
        # Wait a bit if loading is in progress
        time.sleep(0.5)
        return
    
    github_loading_in_progress = True
    
    try:
        username = "StevenMTikas"
        print(f"\n{'='*60}")
        print(f"Loading GitHub repositories for {username} (on-demand)...")
        print(f"{'='*60}")
        
        github_docs = fetch_github_documents(username)
        
        if github_docs:
            print(f"Found {len(github_docs)} GitHub repositories to add")
            add_documents_to_vector_store(github_docs)
            github_documents_loaded = True
            print(f"✓ GitHub repositories loaded successfully ({len(github_docs)} repositories)")
            print(f"{'='*60}\n")
        else:
            print("⚠ No GitHub repositories loaded - this might be due to:")
            print("  - API rate limits (set GITHUB_TOKEN for higher limits)")
            print("  - Network connectivity issues")
            print("  - User has no public repositories")
            github_documents_loaded = True  # Mark as loaded even if empty to avoid retrying
            print(f"{'='*60}\n")
            
    except Exception as e:
        print(f"⚠ Error loading GitHub repositories: {e}")
        import traceback
        traceback.print_exc()
        github_documents_loaded = True  # Mark as loaded to avoid infinite retries
        print(f"{'='*60}\n")
    finally:
        github_loading_in_progress = False

def initialize_vector_store():
    """Initialize the vector store with documents from MeBot folder."""
    global vector_store, llm, retriever, embeddings
    
    try:
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set.\n"
                "Please create a .env file in the project root with:\n"
                "OPENAI_API_KEY=your_api_key_here\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        
        # Clean the API key (remove any whitespace, line breaks)
        api_key = api_key.strip().replace('\n', '').replace('\r', '')
        
        # Validate API key format (should start with sk-)
        if not api_key.startswith("sk-"):
            raise ValueError(
                "Invalid OpenAI API key format. API keys should start with 'sk-'.\n"
                "Please check your .env file and get a valid key from: https://platform.openai.com/api-keys"
            )
        
        # Debug: Show partial key for verification (first 10 and last 4 chars)
        print(f"Using API key: {api_key[:10]}...{api_key[-4:]} (length: {len(api_key)})")
        
        print("Loading documents...")
        documents = load_documents()
        
        if not documents:
            raise ValueError("No documents loaded")
        
        print(f"Total pages loaded: {len(documents)}")
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks")
        
        # Create embeddings and vector store
        print("Creating embeddings and vector store...")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("Vector store created successfully!")
        print(f"Vector store initialized with {len(chunks)} chunks")
        
        # Initialize LLM and retriever
        print("Initializing LLM and retriever...")
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=api_key,
        )
        
        # Create retriever - retrieve more documents to ensure we get all sources
        # This helps when GitHub docs are added later
        retriever = vector_store.as_retriever(search_kwargs={"k": 8})
        
        print("LLM and retriever initialized successfully!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error initializing vector store: {error_msg}")
        
        # Provide helpful error messages for common issues
        if "invalid_api_key" in error_msg or "401" in error_msg or "Incorrect API key" in error_msg:
            print("\n" + "=" * 50)
            print("API KEY ERROR DETECTED")
            print("=" * 50)
            print("Your OpenAI API key is invalid or incorrect.")
            print("\nTo fix this:")
            print("1. Go to https://platform.openai.com/api-keys")
            print("2. Create a new API key or copy your existing one")
            print("3. Update your .env file with:")
            print("   OPENAI_API_KEY=sk-your-actual-key-here")
            print("4. Make sure there are no extra spaces or quotes")
            print("=" * 50)
        
        raise

# Startup is now handled in the lifespan context manager above
# Root route is defined above to serve the HTML page
# Health check is at /health endpoint

@app.post("/api/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Answer a question about Steven Tikas using the document knowledge base."""
    if not llm or not retriever:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Please check server logs."
        )
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Ensure GitHub repositories are loaded (on-demand)
        ensure_github_loaded()
        
        # Retrieve relevant documents
        # Increase k to ensure we get GitHub docs if they exist
        docs = retriever.invoke(request.message)
        
        # Debug: Log what sources we retrieved
        sources = [d.metadata.get('source', 'unknown') for d in docs]
        priorities = [d.metadata.get('priority', 'unknown') for d in docs]
        print(f"Retrieved {len(docs)} documents - Sources: {sources}, Priorities: {priorities}")
        
        # Prioritize documents: primary (bio.txt) > secondary (GitHub) > tertiary (PDFs)
        # Sort by priority: 0=primary, 1=secondary, 2=tertiary, 3=unknown
        def get_priority_value(doc):
            priority = doc.metadata.get('priority', 'unknown')
            priority_map = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}
            return priority_map.get(priority, 3)
        
        docs_sorted = sorted(docs, key=lambda d: (get_priority_value(d), d.metadata.get('source', '')))
        
        # Separate documents by priority
        primary_docs = [d for d in docs_sorted if d.metadata.get('priority') == 'primary']
        secondary_docs = [d for d in docs_sorted if d.metadata.get('priority') == 'secondary']
        tertiary_docs = [d for d in docs_sorted if d.metadata.get('priority') == 'tertiary']
        
        # Format context with clear separation and priority indication
        context_parts = []
        
        if primary_docs:
            bio_content = "\n\n".join([doc.page_content for doc in primary_docs])
            context_parts.append(f"PRIMARY SOURCE (bio.txt - OVERRIDING SOURCE OF TRUTH):\n{bio_content}")
        
        if secondary_docs:
            github_content = "\n\n".join([doc.page_content for doc in secondary_docs])
            context_parts.append(f"SECONDARY SOURCES (GitHub Repositories - Current Projects):\n{github_content}")
        
        if tertiary_docs:
            pdf_content = "\n\n".join([doc.page_content for doc in tertiary_docs])
            context_parts.append(f"TERTIARY SOURCES (Resume and LinkedIn Profile):\n{pdf_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt with emphasis on source priority
        prompt = f"""You are an AI assistant that answers questions about Steven Tikas based on the provided context. You will try to talk him up when possible without lying or being too over the top about it.

IMPORTANT PRIORITY ORDER:
1. PRIMARY SOURCE (bio.txt) is the OVERRIDING SOURCE OF TRUTH. Always prioritize information from bio.txt.
2. SECONDARY SOURCES (GitHub Repositories) contain current project information, descriptions, and README files. Use this for questions about current projects, code repositories, and active development work.
3. TERTIARY SOURCES (Resume and LinkedIn Profile) provide historical and professional background information.

If there are any differences or conflicts between sources, always prioritize and use the information from bio.txt first, then GitHub repositories for current projects, then PDFs for historical information.

Use the following pieces of context to answer the question. If you don't know the answer based on the context, say that you don't have that information, but try to be helpful with what you do know.

Ensure that you are looking at all of the available information before answering any questions to make sure you are providing full information.

Context:
{context}

Question: {request.message}

Answer:"""
        
        # Get answer from LLM
        response = llm.invoke(prompt)
        
        # Extract text from response (handles both string and message object)
        if hasattr(response, 'content'):
            reply = response.content
        elif isinstance(response, str):
            reply = response
        else:
            reply = str(response)
        
        return ChatResponse(reply=reply.strip())
    
    except Exception as e:
        print(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

if __name__ == "__main__":
    # Render uses PORT environment variable, default to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

