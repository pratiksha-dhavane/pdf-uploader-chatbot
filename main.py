import os
import uuid
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Literal
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as BaseModelSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import time
from functools import wraps

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="PDF Chatbot with RAG")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)

# Model for settings with dropdown options
class SettingsModel(BaseModelSettings):
    model: Literal["gemini-1.5-pro", "gemini-1.5-flash"] = Field(
        default="gemini-1.5-pro",
        description="Select the Gemini model to use for generating responses"
    )
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Controls randomness: Lower = more deterministic, Higher = more creative"
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate in the response"
    )

# Global variables
vector_index = None
chat_sessions = {}
current_settings = SettingsModel()  # Initialize with default settings

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)
SESSIONS_FILE = "data/sessions.json"
SETTINGS_FILE = "data/settings.json"

# Load settings from file
def load_settings():
    global current_settings
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                file_content = f.read().strip()
                if file_content:
                    settings_data = json.loads(file_content)
                    current_settings = SettingsModel(**settings_data)
                    logger.info(f"Loaded settings: {current_settings.dict()}")
                else:
                    current_settings = SettingsModel()
                    logger.info("Settings file is empty, using default settings")
        else:
            current_settings = SettingsModel()
            logger.info("No settings file found, using default settings")
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        current_settings = SettingsModel()

# Save settings to file
def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(current_settings.dict(), f, indent=2)
        logger.info(f"Saved settings: {current_settings.dict()}")
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")

# To reduce api call to avoid hitting api limits
def rate_limited(max_per_minute):
    min_interval = 60.0 / max_per_minute
    def decorator(func):
        last_time_called = [0.0]
        @wraps(func)
        def rate_limited_function(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return ret
        return rate_limited_function
    return decorator

# Update LLM settings based on current configuration
def update_llm_settings():
    try:
        Settings.llm = Gemini(
            model=current_settings.model, 
            api_key=GOOGLE_API_KEY, 
            temperature=current_settings.temperature, 
            max_tokens=current_settings.max_tokens
        )
        logger.info(f"Updated LLM settings: {current_settings.dict()}")
    except Exception as e:
        logger.error(f"Error updating LLM settings: {str(e)}")
        raise

# Set Gemini as the default embedding model
try:
    # Settings.embed_model = GeminiEmbedding(
    #     model_name="models/embedding-001", 
    #     api_key=GOOGLE_API_KEY
    # )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    logger.info("Successfully configured embedding model")
except Exception as e:
    logger.error(f"Error configuring embedding model: {str(e)}")
    raise

# Apply rate limiting to your embedding function
@rate_limited(60)  # 60 calls per minute
def get_embedding_with_rate_limit(text):
    return Settings.embed_model.get_text_embedding(text)

# Custom prompt template for RAG
custom_prompt = PromptTemplate(
    "You are an expert assistant specialized in analyzing and answering questions about documents. "
    "You have access to contextual information from uploaded PDF documents.\n\n"
    "Context information from uploaded documents is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Given this context information and not prior knowledge, "
    "answer the query in a clear, detailed and helpful and kind manner. "
    "If the context doesn't contain relevant information, "
    "clearly state that you cannot answer based on the provided documents.\n\n"
    "Query: {query_str}\n\n"
    "Answer:"
)

# Load sessions on startup
def load_sessions():
    global chat_sessions
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, 'r') as f:
                file_content = f.read().strip()
                if file_content:
                    chat_sessions = json.loads(file_content)
                    logger.info(f"Loaded {len(chat_sessions)} sessions from file")
                else:
                    chat_sessions = {}
                    logger.info("Sessions file is empty, starting with empty sessions")
        else:
            chat_sessions = {}
            logger.info("No sessions file found, starting with empty sessions")
    except Exception as e:
        logger.error(f"Error loading sessions: {str(e)}")
        chat_sessions = {}

# Save sessions to file
def save_sessions():
    try:
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(chat_sessions, f, indent=2, default=str)
        logger.info(f"Saved {len(chat_sessions)} sessions to file")
    except Exception as e:
        logger.error(f"Error saving sessions: {str(e)}")

# Load settings and sessions on startup
load_settings()
update_llm_settings()  # Apply settings to LLM
load_sessions()

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    messages: List[Dict[str, Any]]

class SessionCreate(BaseModel):
    name: str

class Session(BaseModel):
    id: str
    name: str
    created_at: str
    messages: List[Dict[str, Any]]

@app.get("/")
def get_home():
    return FileResponse("static/index.html", headers={"Cache-Control": "no-cache"})

@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        # Create upload directory if it doesn't exist
        upload_dir = "uploaded_pdfs"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Clear previous files
        for file in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file_path)
            logger.info(f"Saved file: {file_path}")
        
        # Load documents and create index
        documents = SimpleDirectoryReader(upload_dir).load_data()
        
        # Create FAISS vector store
        dimension = 384
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        global vector_index
        vector_index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            show_progress=True
        )
        
        return JSONResponse(
            content={"message": f"Successfully processed {len(saved_files)} PDF files"},
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        if vector_index is None:
            raise HTTPException(status_code=400, detail="Please upload PDF files first")
        
        # Get or create session
        session_id = chat_request.session_id
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add user message to session
        user_message = {
            "role": "user",
            "content": chat_request.message,
            "timestamp": datetime.now().isoformat()
        }
        chat_sessions[session_id]["messages"].append(user_message)
        
        # Create query engine with custom prompt
        query_engine = vector_index.as_query_engine(
            text_qa_template=custom_prompt,
            similarity_top_k=3
        )
        
        # Query the index
        response = query_engine.query(chat_request.message)
        
        # Add assistant response to session
        assistant_message = {
            "role": "assistant",
            "content": str(response),
            "timestamp": datetime.now().isoformat()
        }
        chat_sessions[session_id]["messages"].append(assistant_message)
        
        # Save sessions
        save_sessions()
        
        return JSONResponse(
            content={
                "response": str(response),
                "session_id": session_id,
                "messages": chat_sessions[session_id]["messages"]
            },
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        # Remove the user message if there was an error
        if session_id in chat_sessions and chat_sessions[session_id]["messages"]:
            chat_sessions[session_id]["messages"].pop()
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.get("/sessions", response_model=List[Session])
async def get_sessions():
    try:
        sessions_list = []
        for session_id, session_data in chat_sessions.items():
            sessions_list.append({
                "id": session_id,
                "name": session_data["name"],
                "created_at": session_data["created_at"],
                "messages": session_data["messages"]
            })
        return sessions_list
    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting sessions: {str(e)}")

@app.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str):
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = chat_sessions[session_id]
        
        # Ensure the session has the required structure
        if not all(key in session_data for key in ["name", "created_at", "messages"]):
            logger.error(f"Session {session_id} has invalid structure: {session_data}")
            raise HTTPException(status_code=500, detail="Session data is corrupted")
        
        return {
            "id": session_id,
            "name": session_data["name"],
            "created_at": session_data["created_at"],
            "messages": session_data["messages"]
        }
    except Exception as e:
        logger.error(f"Error getting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting session: {str(e)}")

@app.post("/sessions", response_model=Session)
async def create_session(session_create: SessionCreate):
    try:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = {
            "name": session_create.name,
            "created_at": datetime.now().isoformat(),
            "messages": []
        }
        
        # Save sessions
        save_sessions()
        
        return {
            "id": session_id,
            "name": chat_sessions[session_id]["name"],
            "created_at": chat_sessions[session_id]["created_at"],
            "messages": chat_sessions[session_id]["messages"]
        }
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        if session_id in chat_sessions:
            del chat_sessions[session_id]
            save_sessions()
            return {"message": "Session deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.post("/settings")
async def update_settings(settings: SettingsModel):
    try:
        global current_settings
        current_settings = settings
        
        # Update the LLM with new settings
        update_llm_settings()
        
        # Save settings to file
        save_settings()
        
        logger.info(f"Settings updated: {settings.dict()}")
        
        return {"message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating settings: {str(e)}")

@app.get("/settings")
async def get_settings():
    return current_settings.dict()

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "sessions_count": len(chat_sessions),
        "vector_index_ready": vector_index is not None
    }

@app.get("/")
async def root():
    return {"message": "PDF Chatbot with RAG API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)