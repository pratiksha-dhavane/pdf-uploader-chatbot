import os
import uuid
import logging
import json
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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

# model for settings
class SettingsModel(BaseModelSettings):
    model: str = "gemini-pro"
    temperature: float = 0.7
    max_tokens: int = 1000

# Set Gemini as the default LLM and embedding model
try:
    Settings.llm = Gemini(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    Settings.embed_model = GeminiEmbedding(
        model_name="models/embedding-001", 
        api_key=GOOGLE_API_KEY
    )
    logger.info("Successfully configured Gemini LLM and embedding models")
except Exception as e:
    logger.error(f"Error configuring Gemini: {str(e)}")
    raise

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

# Global variables
vector_index = None
chat_sessions = {}

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)
SESSIONS_FILE = "data/sessions.json"

# In the load_sessions function, add more detailed error handling:
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

# Format conversation history for the prompt
def format_history(messages):
    history_str = ""
    for msg in messages[-6:]:  # Include last 6 messages for context
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    return history_str

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

# To load settings on startup
def load_settings():
    try:
        if os.path.exists("data/settings.json"):
            with open("data/settings.json", "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        return {}

# Call this function when initializing your app
app_settings = load_settings()

# Load sessions on startup
load_sessions()

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
        dimension = 768  # Gemini embedding dimension
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
        
        # Get settings
        settings = load_settings()
        temperature = settings.get('temperature', 0.7)
        max_tokens = settings.get('max_tokens', 1000)
        
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
        
        # Create query engine with custom prompt and settings
        query_engine = vector_index.as_query_engine(
            text_qa_template=custom_prompt,
            similarity_top_k=3,
            # Apply settings if your LLM supports them
            temperature=temperature,
            max_tokens=max_tokens
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
        # Save settings to a file
        with open("data/settings.json", "w") as f:
            json.dump(settings.dict(), f)
        
        logger.info(f"Settings updated: {settings.dict()}")
        
        return {"message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating settings: {str(e)}")
    
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