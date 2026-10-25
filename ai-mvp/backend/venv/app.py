"""
RAG-enabled Flask backend for Study Buddy.
Endpoints:
- POST /upload - Upload and ingest document
- POST /generate - RAG-enhanced chat
- GET /documents - List ingested documents
- DELETE /documents/<doc_id> - Delete document
- POST /search - Search similar chunks (for debugging)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
import tempfile
import uuid
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Import RAG modules
from ingest import (
    ingest_file,
    list_documents,
    delete_document,
    search_similar,
    get_collection
)

app = Flask(__name__)
CORS(app)

# Initialize Groq client for chat
chat_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Chat model
CHAT_MODEL = "openai/gpt-oss-20b"

# In-memory chat history (per session in production, use Redis)
chat_sessions = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'pptx', 'docx', 'txt', 'md'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_id() -> str:
    """Get or create session ID from request."""
    # In production, use proper session management
    session_id = request.headers.get('X-Session-ID') or request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id


def get_chat_history(session_id: str) -> list:
    """Get chat history for session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    return chat_sessions[session_id]


# System prompts for different modes
SYSTEM_PROMPT_DEFAULT = """You are a study assistant. Explain concepts simply with examples.
Explain it to me like I am five years old. Use analogies and break down complex topics."""

SYSTEM_PROMPT_RAG = """You are a study assistant with access to the user's uploaded documents.
Answer questions based on the provided context from their documents.
- Cite sources using [source] notation (e.g., [slide 3], [page 5])
- If the context doesn't contain the answer, say so honestly
- Explain simply with examples, like you're teaching a 5-year-old
- Use analogies and break down complex topics"""

SYSTEM_PROMPT_EXAM = """You are an exam answer writer. You have access to:
1. Course material (textbook, slides, notes)
2. Previous year papers WITH model answers

Your job: Answer the user's question in the EXACT style, structure, length, and format of the model answers.

Rules:
- Match the paragraph structure, bullet style, diagram references of model answers
- Use the same terminology the professor expects
- Same level of detail (neither too brief nor too verbose)
- If model answers show worked examples, include them
- Cite relevant course material AND similar past paper questions
- Format your answer exactly like a model exam answer"""


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "Study Buddy RAG",
        "endpoints": [
            "POST /upload - Upload document",
            "POST /generate - Chat with RAG",
            "GET /documents - List documents",
            "DELETE /documents/<doc_id> - Delete document",
            "POST /search - Search chunks"
        ]
    })


@app.route("/upload", methods=["POST"])
def upload_document():
    """Upload and ingest a document."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": f"File too large. Max {MAX_FILE_SIZE // (1024*1024)}MB"}), 400

    # Save to temp file
    filename = secure_filename(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        # Ingest document
        result = ingest_file(temp_path)

        # Clean up temp file
        os.unlink(temp_path)

        if result["status"] == "success":
            return jsonify({
                "success": True,
                "doc_id": result["doc_id"],
                "filename": result["filename"],
                "chunks": result["chunks"],
                "pages_slides": result["pages_slides"]
            })
        else:
            return jsonify({
                "success": False,
                "message": result.get("message", "Unknown error")
            }), 400

    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": str(e)}), 500


@app.route("/documents", methods=["GET"])
def get_documents():
    """List all ingested documents."""
    try:
        docs = list_documents()
        return jsonify({"documents": docs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    """Delete a document."""
    try:
        result = delete_document(doc_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/search", methods=["POST"])
def search_chunks():
    """Search for similar chunks (debugging)."""
    data = request.json
    query = data.get("query", "")
    top_k = data.get("top_k", 5)
    doc_id = data.get("doc_id")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        results = search_similar(query, top_k=top_k, doc_id=doc_id)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """RAG-enhanced chat generation."""
    session_id = get_session_id()
    chat_history = get_chat_history(session_id)

    data = request.json
    user_input = data.get("input", "").strip()
    mode = data.get("mode", "rag")  # default, rag, exam
    doc_id = data.get("doc_id")  # Optional: restrict to specific document

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Add user message to history
    chat_history.append({"role": "user", "content": user_input})

    try:
        # Select system prompt based on mode
        if mode == "exam":
            system_prompt = SYSTEM_PROMPT_EXAM
        elif mode == "rag":
            system_prompt = SYSTEM_PROMPT_RAG
        else:
            system_prompt = SYSTEM_PROMPT_DEFAULT

        # Retrieve relevant context if in RAG or Exam mode
        context = ""
        sources = []

        if mode in ["rag", "exam"]:
            search_results = search_similar(user_input, top_k=5, doc_id=doc_id)
            if search_results:
                context_parts = []
                for i, r in enumerate(search_results):
                    meta = r["metadata"]
                    source_label = ""
                    if "page" in meta:
                        source_label = f"page {meta['page']}"
                    elif "slide" in meta:
                        source_label = f"slide {meta['slide']}"
                    elif "paragraph" in meta:
                        source_label = f"para {meta['paragraph']}"
                    else:
                        source_label = f"chunk {i+1}"

                    context_parts.append(f"[{source_label}] {r['text']}")
                    sources.append({
                        "label": source_label,
                        "filename": meta.get("filename"),
                        "text_preview": r["text"][:200]
                    })

                context = "\n\n".join(context_parts)

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add context if available
        if context:
            context_message = f"Relevant context from your documents:\n{context}"
            messages.append({"role": "system", "content": context_message})

        # Add chat history (keep last 10 messages to avoid token limits)
        messages.extend(chat_history[-10:])

        # Generate response
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )

        ai_reply = response.choices[0].message.content

        # Add AI response to history
        chat_history.append({"role": "assistant", "content": ai_reply})

        # Trim history if too long (keep last 20 messages)
        if len(chat_history) > 20:
            chat_sessions[session_id] = chat_history[-20:]

        return jsonify({
            "response": ai_reply,
            "sources": sources,
            "mode": mode,
            "session_id": session_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/session/new", methods=["POST"])
def new_session():
    """Start a new chat session."""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = []
    return jsonify({"session_id": session_id})


if __name__ == "__main__":
    app.run(debug=True, port=5000)