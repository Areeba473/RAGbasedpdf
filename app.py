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
    raise ValueError("❌ GROQ_API_KEY not found. Add it in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

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
Answer strictly using the provided context.
If the answer is not present, say:
"I don't know based on the document."

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

    return f"✅ PDF indexed successfully ({len(doc_chunks)} chunks)."


def answer_question(question):
    if faiss_index is None:
        return "⚠️ Please upload a PDF first."

    context = retrieve_context(question)
    return ask_groq(context, question)

# ================= UI (ZINC / BLACK / GREEN) =================

css = """
body {
    background: linear-gradient(135deg, #09090B, #18181B);
    color: #E4E4E7;
    font-family: 'Inter', system-ui, sans-serif;
}
h1, label {
    color: #E4E4E7 !important;
    font-weight: 600;
}
p {
    color: #A1A1AA !important;
}
textarea, input, select {
    background-color: #18181B !important;
    color: #E4E4E7 !important;
    border-radius: 12px !important;
    border: 1px solid #27272A !important;
    padding: 10px !important;
}
textarea:focus, input:focus, select:focus {
    outline: none !important;
    border-color: #22C55E !important;
    box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.5);
}
button {
    background: linear-gradient(90deg, #16A34A, #22C55E) !important;
    color: #052E16 !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    padding: 12px !important;
    border: none !important;
}
button:hover {
    box-shadow: 0 0 18px rgba(34, 197, 94, 0.45);
    transform: translateY(-1px);
}
.gradio-container {
    max-width: 900px;
    margin: auto;
}
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #09090B;
}
::-webkit-scrollbar-thumb {
    background: #22C55E;
    border-radius: 10px;
}
"""

with gr.Blocks() as demo:
    gr.Markdown("""
    <h1 style="text-align:center; color:#22C55E;">
    📄 RAG-Based PDF Chatbot
    </h1>
    <p style="text-align:center;">
    </p>
    """)

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
    ssr_mode=False
)
