import os
from openai import OpenAI
import pdfplumber
import docx
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import subprocess
import secrets

# Generate a secure random secret key

load_dotenv()  # Load API key from .env file

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'xlsm', 'xlsx'}

client = OpenAI()
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload folder exists

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_doc_to_docx(doc_file):
    try:
        # Convert .doc to .docx using unoconv
        output_file = doc_file.rsplit('.', 1)[0] + '.docx'
        subprocess.run(["unoconv", "-f", "docx", doc_file], check=True)
        return output_file
    except Exception as e:
        return str(e)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text(file_path):
    """Extracts text from uploaded files"""
    ext = file_path.rsplit('.', 1)[1].lower()
    text = ""

    try:
        if ext == "pdf":
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        elif ext == "docx":
            text = extract_text_from_docx(file_path)
        
        elif ext == "doc":
            docx_file = convert_doc_to_docx(file_path)  # Convert .doc to .docx
            text = extract_text_from_docx(docx_file)
        
        elif ext in {"xlsm", "xlsx"}:
            df = pd.read_excel(file_path, sheet_name=None)  # Load all sheets
            text = "\n".join([
                "\n".join(map(str, sheet.fillna("").to_numpy().flatten()))  # Fill NaN with empty string
                for _, sheet in df.items()
            ])
        
        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        return str(e)  # Return error message if extraction fails
    
    return text if text else "No text extracted"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file uploads and stores extracted text in session"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure the directory exists
    file.save(file_path)

    text = extract_text(file_path)
    if not text.strip():
        return jsonify({"error": "Could not extract text from file"}), 400

    # Store document text in session
    session["document_text"] = text[:4000]  # Keep first 4000 chars (API limit)

    return jsonify({"response": "File uploaded and analyzed. You can now ask questions about it."})

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat messages and includes uploaded document in context"""
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Retrieve document text from session
    document_text = session.get("document_text", "")

    # Ensure the AI knows it should reference the uploaded document
    system_message = (
        "You are an AI assistant that can answer both general questions and questions based on an uploaded document. "
        "If the user's question relates to the document, prioritize using the document's content. "
        "Otherwise, use your general knowledge."
    )

    context_message = f"Here is the uploaded document:\n\n{document_text[:4000]}" if document_text else ""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "system", "content": context_message},
        {"role": "user", "content": user_message}
    ]

    messages = [msg for msg in messages if msg["content"].strip()]  # Remove empty messages

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500

    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
