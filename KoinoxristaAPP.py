import streamlit as st
from pdf_generator import create_pdf
from rag_assistant import ask_rag, load_models

# Page Configuration
st.set_page_config(page_title="Building Management App", layout="wide")

# 1. LOAD MODELS (Cached)
@st.cache_resource
def get_models():
    return load_models()

embedding_model, pipe = get_models()

# 2. SESSION STATE FOR CHAT
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "system", 
            "content": (
                "Είσαι ένας χρήσιμος βοηθός διαχείρισης κτηρίου. "
                "Απάντησε στην ερώτηση του χρήστη αποκλειστικά και μόνο με βάση τις πληροφορίες που σου δίνονται, στα Ελληνικά."
            )
        }
    ]

# --- UI LAYOUT ---
st.title("🏢 Διαχείριση Κοινοχρήστων & RAG Assistant")

col1, col2 = st.columns([1, 1])

# ΑΡΙΣΤΕΡΗ ΣΤΗΛΗ: Υπολογισμός & Έκδοση PDF
with col1:
    st.header("📄 Έκδοση Κοινοχρήστων")
    with st.form("data_form"):
        period = st.text_input("Περίοδος", "16/02/26 - 15/03/26")
        reuma = st.number_input("Σύνολο Ρεύματος (€)", min_value=0.0, format="%.2f")
        nero = st.number_input("Σύνολο Νερού (€)", min_value=0.0, format="%.2f")
        episkeves = st.number_input("Σύνολο Επισκευών (€)", min_value=0.0, format="%.2f")
        submit_button = st.form_submit_button("Υπολογισμός")

    if submit_button:
        st.success("Το PDF δημιουργήθηκε επιτυχώς!")
        pdf_bytes = create_pdf(period, reuma, nero, episkeves)

        st.download_button(
            label="📥 Λήψη PDF",
            data=bytes(pdf_bytes),
            file_name=f"koinoxrista_{period}.pdf",
            mime="application/pdf"
        )

# ΔΕΞΙΑ ΣΤΗΛΗ: RAG Chatbot
with col2:
    st.header("🤖 Βοηθός RAG")
    
    # Προβολή Ιστορικού (αγνοώντας το system message)
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat Input
    if user_input := st.chat_input("Ρωτήστε για τους κανόνες κοινοχρήστων..."):
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Ανάλυση..."):
                response = ask_rag(
                    user_query=user_input,
                    history=st.session_state.chat_history,
                    pipe=pipe
                )
                st.write(response)