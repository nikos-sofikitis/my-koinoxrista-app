import os
import torch
from threading import Thread
from transformers import pipeline, TextIteratorStreamer

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- GLOBAL ΜΕΤΑΒΛΗΤΕΣ ---
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

    # 3. HuggingFace Embeddings
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

    # Β. Φόρτωση Local LLM Pipeline (Qwen 3.5 0.8B)
    device_id = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3.5-0.8B",
        model_kwargs={"cache_dir": CACHE_DIR, "torch_dtype": "auto"},
        device=device_id
    )

    return vectorstore, pipe


def ask_rag_stream(user_query, history, max_history_turns=3):
    """Generator Function που επιστρέφει tokens σε real-time (Streaming)."""
    global vectorstore, pipe

    context = ""
    # A. Similarity Search στα PDFs
    if vectorstore is not None:
        retrieved_docs = vectorstore.similarity_search(user_query, k=4)
        context = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])

    # B. Sliding Window History Management
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

    # D. Setup Streamer & Threading
    tokenizer = pipe.tokenizer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        text_inputs=prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        streamer=streamer
    )

    # Εκτέλεση του pipeline σε ξεχωριστό thread για να μην μπλοκάρει το generator
    thread = Thread(target=pipe, kwargs=generation_kwargs)
    thread.start()

    # E. Yield tokens καθώς παράγονται
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text