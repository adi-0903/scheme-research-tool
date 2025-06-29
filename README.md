# 🧠 Haqdarshak Scheme Research Tool (Local, OpenAI-Free)

This is a **Streamlit web app** designed to automate scheme research for Haqdarshak. It allows users to input government scheme URLs (including PDFs), extract and summarize content, and **ask questions** using a local language model — **no OpenAI API key required**.

---

## ✅ Features

- 🔗 Load articles from URLs (web pages or PDFs)
- 📄 Extract and chunk content
- 🧠 Create embeddings using HuggingFace
- ⚡ Store and search with FAISS for fast Q&A
- 💬 Ask questions and get answers with source references
- 🚫 No OpenAI — runs entirely on free or local models

---

## 🧰 Technologies Used

- `Streamlit` for the web interface
- `LangChain` for document and chain management
- `HuggingFace Transformers` for free local language models
- `FAISS` for vector similarity search
- `PyMuPDF (fitz)` for PDF parsing
- `venv` for isolated Python environments

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/adi-0903/scheme-research-tool.git
cd scheme-research-tool

2. Set Up Virtual Environment

python -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Run the App


streamlit run main.py

```

📂 File Structure


haqdarshak-scheme-research/

│

├── main.py               # Full app in one file

├── faiss_store_openai.pkl (auto-generated)

├── README.md             # This file

├── requirements.txt      # Python dependencies

└── venv/                 # Your virtual environment (excluded from Git)



✍️ Usage


Paste one or more scheme article URLs (can be PDFs or web pages).


Click Process URLs to build the index.


Type your question in natural language.


Get relevant answers + source links!


# 🛠️ Notes


Internet is required for downloading models the first time.


If models are too slow, try switching to smaller ones (e.g., google/flan-t5-small).


Index is saved as faiss_store_openai.pkl for reuse.
