import os
import gradio as gr
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Add it to environment variables.")

client = Groq(api_key=GROQ_API_KEY)

EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

# Global storage (HF free-tier friendly)
faiss_index = None
doc_chunks = []

# ================= PDF PROCESSING =================

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# ================= FAISS =================

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_context(query, k=4):
    query_embedding = embedder.encode([query])
    _, indices = faiss_index.search(query_embedding, k)
    return "\n\n".join([doc_chunks[i] for i in indices[0]])


# ================= GROQ LLM =================

def ask_groq(context, question):
    prompt = f"""
You are a helpful AI assistant.
Answer the question strictly using the provided context.
If the answer is not present, say "I don't know based on the document."

Context:
{context}

Question:
{question}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    return completion.choices[0].message.content


# ================= MAIN LOGIC =================

def process_pdf(pdf):
    global faiss_index, doc_chunks

    text = load_pdf(pdf)
    if not text.strip():
        return "❌ No readable text found in PDF."

    doc_chunks = chunk_text(text)
    faiss_index = create_faiss_index(doc_chunks)

    return f"✅ PDF processed successfully!\nChunks created: {len(doc_chunks)}"


def answer_question(question):
    if faiss_index is None:
        return "⚠️ Please upload a PDF first."

    context = retrieve_context(question)
    return ask_groq(context, question)


# ================= UI =================

css = """
body {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
}
h1, label {
    color: #E5E7EB !important;
}
textarea, input {
    background-color: #1F2937 !important;
    color: white !important;
    border-radius: 10px !important;
}
button {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 14px !important;
}
button:hover {
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.6);
}
"""

with gr.Blocks() as demo:
    gr.Markdown(
        "<h1 style='text-align:center'>📄 RAG-Based PDF Chatbot</h1>"
        "<p style='text-align:center;color:#CBD5F5'>Groq · FAISS · Gradio</p>"
    )

    with gr.Group():
        pdf = gr.File(label="Upload PDF")
        status = gr.Textbox(label="Status", interactive=False)
        pdf.upload(process_pdf, pdf, status)

    with gr.Group():
        question = gr.Textbox(label="Ask a Question")
        answer = gr.Textbox(lines=8, label="Answer")

    gr.Button("🔍 Ask").click(answer_question, question, answer)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    css=css,
    ssr_mode=False   # 🔥 HF crash-safe
)
