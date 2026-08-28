# 🏢 My Koinoxrista App (Smart Building Expense Manager & RAG Assistant)

A modular Python & Streamlit application designed for automated building expense calculations, PDF report generation, and intelligent document querying via an Agentic RAG pipeline.

---

## 🌟 Key Features

* **🧮 Automated Expense Allocation:** Precise calculation of building management expenses per apartment based on specified millimes (χιλιοστά).
* **📄 PDF Report Generation:** Dynamic generation and export of individual and summary expense notices (`fpdf2`).
* **🧠 Smart RAG System (FAISS + LangChain):** Natural language QA interface for building regulations and policy documents using semantic search.
* **🔄 Fallback Strategy:** Intelligent prompt routing that prioritizes retrieved context from FAISS vector storage while seamlessly falling back to general LLM knowledge when local context is missing.

---

## 🏗️ Architecture & RAG Pipeline

---

## 🛠️ Tech Stack

* **Frontend / UI:** Streamlit
* **RAG Framework:** LangChain
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings & LLM:** Hugging Face Inference API (`Qwen/Qwen2.5-72B-Instruct`)
* **Document Processing:** PyPDF / LangChain Document Loaders & Splitters
* **PDF Generation:** `fpdf2`
* **Environment Management:** `python-dotenv`

---

## 📂 Project Structure

```text
my-koinoxrista-app/
├── KoinoxristaAPP.py     # Main Streamlit web application & interface
├── pdf_generator.py      # PDF document creation and layout utilities
├── requirements.txt      # Project Python dependencies
├── .env                  # Environment variables (Hugging Face API tokens)
├── docs/                 # Directory containing building regulation PDFs & notices
└── faiss_index/          # Persisted FAISS vector store index files

🚀 Getting Started
1. Prerequisites
Python 3.10+

Hugging Face API Token

2. Installation
Clone the repository and install dependencies:

Bash
git clone [https://github.com/your-username/my-koinoxrista-app.git](https://github.com/your-username/my-koinoxrista-app.git)
cd my-koinoxrista-app
pip install -r requirements.txt

3. Environment Setup
Create a .env file in the root directory:


HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here

4. Running the Application
Launch the Streamlit dashboard:

Bash
streamlit run KoinoxristaAPP.py

📝 Usage
Calculate Expenses: Input monthly bill totals and generate allocation reports instantly.

Document Indexing: Place building regulations or past notices inside the docs/ folder to automatically build/update the local FAISS index.

Ask Questions: Use the chat panel to ask questions regarding building rules (e.g., "Πώς κατανέμονται οι δαπάνες αυτόνομης θέρμανσης;").
