import os
from openai import OpenAI
import pdfplumber
import docx
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()  # Load API key from .env file

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session storage
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'xlsm', 'xlsx'}

client = OpenAI()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path):
    """Extracts text from uploaded files"""
    ext = file_path.rsplit('.', 1)[1].lower()
    text = ""

    if ext == "pdf":
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    
    elif ext == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    
    elif ext in {"xlsm", "xlsx"}:
        df = pd.read_excel(file_path, sheet_name=None)  # Load all sheets
        text = "\n".join([
            "\n".join(map(str, sheet.replace({np.nan: ""}).to_numpy().flatten()))  
            for _, sheet in df.items()
        ])
    
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    return text

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
        "You have access to a document uploaded by the user. Here is the content:\n\n"
        f"{document_text}\n\n"
        "Now answer the user's question based on this document."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    reply = response.choices[0].message.content
    return jsonify({"response": reply})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
