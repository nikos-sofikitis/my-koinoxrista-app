import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- GLOBAL METABΛΗΤΕΣ ---
vectorstore = None
pipe = None

DOCS_DIR = "docs"
CACHE_DIR = "./HF-CACHE"


def build_or_load_vectorstore():
    """Διαβάζει όλα τα PDFs από τον φάκελο docs/ και φτιάχνει τη Vector Database (FAISS)."""
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # 1. Loading PDFs
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    raw_documents = loader.load()

    if not raw_documents:
        print(f"⚠️ Προειδοποίηση: Το τοπικό docs/ δεν έχει PDFs ακόμα.")
        return None

    # 2. Chunking (Τεμαχισμός)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(raw_documents)

    # 3. HuggingFace Embeddings (Ίδιο μοντέλο που χρησιμοποιούσες)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder=CACHE_DIR
    )

    # 4. Αποθήκευση στο FAISS Vector Store
    v_store = FAISS.from_documents(docs, embeddings)
    return v_store


def load_models():
    """Φόρτωση των μοντέλων κατά την εκκίνηση της εφαρμογής."""
    global vectorstore, pipe
    
    # Α. Φόρτωση/Δημιουργία Vector Database από τα PDFs
    vectorstore = build_or_load_vectorstore()

    # Β. Φόρτωση LLM Pipeline (Qwen 3.5 0.8B)
    device_id = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3.5-0.8B",
        model_kwargs={"cache_dir": CACHE_DIR, "torch_dtype": "auto"},
        device=device_id
    )

    return vectorstore, pipe


def ask_rag(user_query, history, pipe, max_history_turns=3):
    """Production RAG Function με LangChain Retrieval & Memory Window."""
    global vectorstore

    context = ""
    # A. Retrieval από τα PDFs (αν υπάρχει vectorstore)
    if vectorstore is not None:
        # Παίρνουμε τα 4 πιο σχετικά chunks από τα PDFs
        retrieved_docs = vectorstore.similarity_search(user_query, k=4)
        context = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])

    # B. Sliding Window History Management
    window_size = max_history_turns * 2
    
    if history and history[0].get("role") == "system":
        system_msg = history[0]
        past_turns = history[1:]
    else:
        system_msg = {
            "role": "system", 
            "content": "Είσαι ένας χρήσιμος βοηθός διαχείρισης κτηρίου. Απάντησε στην ερώτηση αποκλειστικά στα Ελληνικά με βάση τις πληροφορίες που σου δίνονται."
        }
        past_turns = history

    recent_history = past_turns[-window_size:] if len(past_turns) > window_size else past_turns
    active_messages = [system_msg] + [msg.copy() for msg in recent_history]

    # C. Dynamic Context Injection
    formatted_user_prompt = (
        f"Απάντησε στα Ελληνικά χρησιμοποιώντας αποκλειστικά τις παρακάτω πληροφορίες από τα έγγραφα.\n\n"
        f"Πληροφορίες από PDFs:\n{context if context else 'Δεν βρέθηκαν διαθέσιμα έγγραφα.'}\n\n"
        f"Ερώτηση: {user_query}"
    )

    active_messages.append({"role": "user", "content": formatted_user_prompt})

    # D. Generation
    output = pipe(
        active_messages, 
        max_new_tokens=150, 
        do_sample=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False
    )

    # E. Response Extraction
    raw_response = output[0]["generated_text"]
    if isinstance(raw_response, list):
        response_text = raw_response[-1]["content"].strip()
    else:
        response_text = raw_response.strip()

    # F. History Update
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response_text})
    
    return response_text