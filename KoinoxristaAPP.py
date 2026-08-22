import os
from dotenv import load_dotenv

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

# Φόρτωση API Key από το .env αρχείο
load_dotenv()

# --- GLOBAL METABΛΗΤΕΣ ---
vectorstore = None
llm_chain = None

DOCS_DIR = "docs"


def build_or_load_vectorstore():
    """Διαβάζει τα PDFs από τον φάκελο docs/ και φτιάχνει τη Vector Database (FAISS)."""
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # 1. Loading PDFs
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    raw_documents = loader.load()

    if not raw_documents:
        print(f"⚠️ Προειδοποίηση: Ο φάκελος '{DOCS_DIR}' είναι άδειος.")
        return None

    # 2. Chunking (Τεμαχισμός)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(raw_documents)

    # 3. Fast Local Embeddings (Sentence Transformers)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. Vector Store creation
    return FAISS.from_documents(docs, embeddings)


def load_models():
    """Αρχικοποίηση του Vector Store και σύνδεση με το Hugging Face API."""
    global vectorstore, llm_chain

    # Α. Φόρτωση Vector Store
    vectorstore = build_or_load_vectorstore()

    # Β. Σύνδεση με το Open-Source LLM μέσω Serverless API
    # Χρησιμοποιούμε το Qwen/Qwen2.5-72B-Instruct που είναι κορυφαίο στα Ελληνικά
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        max_new_tokens=256,
        temperature=0.1,
        huggingfacehub_api_token=hf_api_token
    )

    return vectorstore, llm


def ask_rag(user_query, history, llm, max_history_turns=3):
    """Production RAG Function με LangChain Similarity Search & Cloud API Inference."""
    global vectorstore

    context = ""
    # A. Retrieval από τα PDFs
    if vectorstore is not None:
        retrieved_docs = vectorstore.similarity_search(user_query, k=4)
        context = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])

    # B. History Management (Sliding Window)
    window_size = max_history_turns * 2
    recent_history = history[-window_size:] if history else []
    
    history_str = ""
    for msg in recent_history:
        role = "Χρήστης" if msg["role"] == "user" else "Βοηθός"
        history_str += f"{role}: {msg['content']}\n"

    # C. System Prompt Construction
    prompt = (
        "<|im_start|>system\n"
        "Είσαι ένας εξειδικευμένος βοηθός διαχείρισης κτηρίου. "
        "Απάντησε στην ερώτηση του χρήστη αποκλειστικά στα Ελληνικά, χρησιμοποιώντας τις πληροφορίες από τα έγγραφα.\n\n"
        f"--- ΠΛΗΡΟΦΟΡΙΕΣ ΑΠΟ PDFs ---\n{context if context else 'Δεν υπάρχουν διαθέσιμα έγγραφα.'}\n---------------------------\n"
        f"<|im_end|>\n"
        f"{history_str}"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # D. API Call (Εκτελείται στο Cloud, όχι τοπικά!)
    try:
        response_text = llm.invoke(prompt).strip()
    except Exception as e:
        response_text = f"Σφάλμα κατά την κλήση του API: {str(e)}"

    # E. Ενημέρωση History
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response_text})

    return response_text