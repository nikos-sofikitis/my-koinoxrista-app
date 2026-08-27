# 1. Load environment variables from a .env file
import os
from dotenv import load_dotenv

# --- LANGCHAIN COMMUNITY IMPORTS ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS

# --- LANGCHAIN CORE IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# --- OTHER INTEGRATIONS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()  # Load environment variables from .env file
docs_dir = 'docs'  # Directory containing PDF files
FAISS_INDEX_PATH = 'faiss_index'  # Path to save/load FAISS index


def build_or_load_vectorstore():
    # 1. Load a light embedding model from HuggingFace that supports multilingual embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ) 

    # 2. Check if FAISS index already exists
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading Existing FAISS index from disk...")
        return FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("Building FAISS index from PDF documents...")

    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    # 3. Loading PDFs
    loader = PyPDFDirectoryLoader(docs_dir)
    try:
        raw_documents = loader.load()
    except Exception as e:
        print(f"⚠️ Σφάλμα κατά τη φόρτωση των PDFs: {e}")
        return None

    if not raw_documents:
        print(f"⚠️ Προειδοποίηση: Ο φάκελος '{docs_dir}' είναι άδειος ή δεν βρέθηκαν PDF αρχεία.")
        return None

    # CHUNKING
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    docs = text_splitter.split_documents(raw_documents)

    # CREATE VECTOR STORE
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)  # Save the FAISS index to disk

    return vectorstore


def load_models():
    # INITIALIZE VECTORSTORE
    vectorstore = build_or_load_vectorstore()

    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.2,
        huggingfacehub_api_token=hf_api_token
    )

    chat_llm = ChatHuggingFace(llm=llm_endpoint)

    return vectorstore, chat_llm


def ask_rag(user_query, history, vectorstore, chat_llm, max_history_turns=3):
    # Production RAG using LCEL Chain and StrOutputParser
    context = ""

    if vectorstore is not None:
        try:
            retrieved_docs = vectorstore.similarity_search(user_query, k=4)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs]) 
        except Exception as e:
            print(f"⚠️ Σφάλμα κατά την ανάκτηση εγγράφων: {e}")
            context = ""

    if not context:
        context = "Δεν βρέθηκαν σχετικές πληροφορίες στα τοπικά έγγραφα της πολυκατοικίας."

    # History Management
    window_size = max_history_turns * 2
    recent_history = history[-window_size:] if history else []

    chat_messages = []
    for msg in recent_history:
        if msg["role"] == "user":
            chat_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_messages.append(AIMessage(content=msg["content"]))

    # --- HYBRID RAG SYSTEM PROMPT ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Είσαι ένας εξειδικευμένος και έμπειρος βοηθός διαχείρισης κτηρίου, κοινοχρήστων και λογιστικών/πρακτικών θεμάτων πολυκατοικίας.\n"
         "Απάντησε στην ερώτηση του χρήστη πάντα στα Ελληνικά, ακολουθώντας τους παρακάτω κανόνες:\n\n"
         "1. ΕΑΝ η ερώτηση αφορά συγκεκριμένα στοιχεία της πολυκατοικίας (π.χ. συμβόλαια ενοίκων, ονόματα, συγκεκριμένες χρεώσεις/ποσά) και η απάντηση υπάρχει στα παρακάτω εγγραφα:\n"
         "   - Απάντησε στηριζόμενος με ακρίβεια στα έγγραφα.\n\n"
         "2. ΕΑΝ η ερώτηση αφορά γενικές λογιστικές/νομικές απορίες, επίλυση διαφωνιών μεταξύ ενοίκων, ή πρακτικές προτάσεις για προβλήματα πολυκατοικίας που ΔΕΝ υπάρχουν στα έγγραφα:\n"
         "   - ΜΗΝ πεις ότι δεν γνωρίζεις.\n"
         "   - Χρησιμοποίησε τη γενική σου γνώση για να δώσεις μια χρηστική, δομημένη και επαγγελματική συμβουλή ή λύση.\n\n"
         "--- ΠΛΗΡΟΦΟΡΙΕΣ ΑΠΟ ΕΓΓΡΑΦΑ (CONTEXT) ---\n{context}\n-----------------------------------------"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    # Execution via LCEL CHAIN
    try:
        chain = prompt | chat_llm | StrOutputParser()
        response_text = chain.invoke({
            "context": context, 
            "history": chat_messages, 
            "question": user_query
        })
    except Exception as e:
        print(f"\n❌ ΑΚΡΙΒΕΣ ΣΦΑΛΜΑ LLM: {repr(e)}\n")
        response_text = f"⚠️ Σφάλμα LLM: {e}"

    return response_text