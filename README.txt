# 🏢 Building Management App & RAG Assistant

An interactive, end-to-end Python web application built with **Streamlit** for automated building expense allocation, dynamic PDF statement generation, and intelligent tenant query resolution using an **Artificial Intelligence RAG (Retrieval-Augmented Generation)** system.

---

## 🌟 Key Features

* **Automated Expense Allocation:** Calculates individual tenant share for electricity (ΗΡΩΝ), water, cleaning, elevator maintenance, and building repair costs based on predetermined percentages and flat rates.
* **PDF Report Generation:** Dynamically generates and exports clean, downloadable monthly expense statements in PDF format.
* **AI-Powered RAG Chatbot:** An intelligent assistant that answers tenant questions regarding building rules, percentages, and policies exclusively in Greek.
* **Semantic Search:** Uses vector embeddings to retrieve relevant knowledge base context and minimize model hallucinations.
* **Streamlit Caching:** Optimized execution flow using `@st.cache_resource` and `@st.cache_data` to minimize model loading latency and ensure high performance.

---

## 🛠️ Tech Stack

* **Frontend & Web Framework:** Streamlit
* **AI / NLP / RAG Architecture:**
  * Hugging Face Transformers Pipeline
  * Sentence Transformers (Text Embeddings)
  * Vector Search & Cosine Similarity
  * Context-Constrained Prompt Engineering
* **Document Processing:** PDF Generator Engine
* **Version Control & Management:** Git, GitHub, Python Virtual Environment (`venv`)

---

## 📂 Project Structure

```text
my-koinoxrista-app/
├── KoinoxristaAPP.py    # Main Streamlit UI and app layout
├── rag_assistant.py     # RAG embedding retrieval & LLM generation pipelines
├── pdf_generator.py     # PDF creation module for expense reporting
├── requirements.txt     # Python dependency list
├── .gitignore           # Git exclusion rules
└── README.md            # Project documentation