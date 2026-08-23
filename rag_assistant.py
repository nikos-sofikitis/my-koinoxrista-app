import os
from dotenv import load_dotenv

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Φόρτωση API Key από το .env αρχείο
load_dotenv()

# --- GLOBAL METABΛΗΤΕΣ ---
vectorstore = None
DOCS_DIR = "docs"


def build_or_load_vectorstore():
    """Διαβάζει τα PDFs από τον φάκελο docs/ και φτιάχνει τη Vector Database (FAISS)."""
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # 1. Loading PDFs
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    try:
        raw_documents = loader.load()
    except Exception as e:
        print(f"⚠️ Σφάλμα κατά τη φόρτωση των PDFs: {e}")
        return None

    if not raw_documents:
        print(f"⚠️ Προειδοποίηση: Ο φάκελος '{DOCS_DIR}' είναι άδειος ή δεν βρέθηκαν PDF αρχεία.")
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
    global vectorstore

    # Α. Φόρτωση Vector Store
    vectorstore = build_or_load_vectorstore()

    # Β. Σύνδεση με το Open-Source LLM μέσω Serverless API
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=hf_api_token
    )
    
    # Χρήση του ChatHuggingFace wrapper για σωστό chat formatting
    chat_llm = ChatHuggingFace(llm=llm_endpoint)

    return vectorstore, chat_llm


def ask_rag(user_query, history, llm, max_history_turns=3):
    """Production RAG Function με LangChain Similarity Search & Cloud API Inference."""
    global vectorstore

    context = ""
    # A. Retrieval από τα PDFs
    if vectorstore is not None:
        try:
            retrieved_docs = vectorstore.similarity_search(user_query, k=4)
            context = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])
        except Exception as e:
            print(f"⚠️ Σφάλμα κατά το similarity search: {e}")
            context = ""

    if not context:
        context = "Δεν υπάρχουν διαθέσιμα σχετικά έγγραφα."

    # B. History Management (Convert list of dicts to LangChain BaseMessages)
    window_size = max_history_turns * 2
    recent_history = history[-window_size:] if history else []
    
    chat_messages = []
    for msg in recent_history:
        if msg["role"] == "user":
            chat_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_messages.append(AIMessage(content=msg["content"]))

    # C. System Prompt & Chat Template Construction
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Είσαι ένας εξειδικευμένος βοηθός διαχείρισης κτηρίου και κοινοχρήστων.\n"
         "Απάντησε στην ερώτηση του χρήστη αποκλειστικά στα Ελληνικά, χρησιμοποιώντας τις πληροφορίες από τα έγγραφα.\n\n"
         "--- ΠΛΗΡΟΦΟΡΙΕΣ ΑΠΟ PDFs ---\n{context}\n---------------------------"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    # D. Execution via LCEL Chain
    try:
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "history": chat_messages,
            "question": user_query
        })
        response_text = response.content.strip()
    except Exception as e:
        response_text = f"Σφάλμα κατά την κλήση του API: {str(e)}"

    return response_text