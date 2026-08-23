import streamlit as st
from dotenv import load_dotenv

# Εισαγωγή συναρτήσεων από το rag_assistant.py
from rag_assistant import load_models, ask_rag

# Φόρτωση μεταβλητών περιβάλλοντος (.env)
load_dotenv()

# --- ΡΥΘΜΙΣΗ ΣΕΛΙΔΑΣ STREAMLIT ---
st.set_page_config(
    page_title="Διαχείριση Πολυκατοικίας - Φιλικών 35",
    page_icon="🏢",
    layout="wide"
)

# --- INITIALIZATION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

if "models_loaded" not in st.session_state:
    with st.spinner("📦 Φόρτωση μοντέλων και εγγράφων PDF..."):
        st.session_state.vectorstore, st.session_state.llm = load_models()
    st.session_state.models_loaded = True

# --- SIDEBAR: ΣΤΟΙΧΕΙΑ & PDF GENERATOR ---
with st.sidebar:
    st.title("🏢 Φιλικών 35, Περιστέρι")
    st.caption("Διαχειριστής: Σωτήρης Σοφικίτης")
    st.divider()

    st.subheader("📄 Έκδοση Κοινοχρήστων")
    st.write("Συμπλήρωσε τα στοιχεία του μήνα για υπολογισμό.")

    month_input = st.text_input("Μήνας / Έτος", value="Μάρτιος 2026")
    hron_bill = st.number_input("Λογαριασμός ΗΡΩΝ (€)", min_value=0.0, value=120.0, step=5.0)
    water_bill = st.number_input("Λογαριασμός Νερού (€) - [0 αν δεν υπάρχει]", min_value=0.0, value=0.0, step=5.0)

    if st.button("🚀 Υπολογισμός Κοινοχρήστων", use_container_width=True):
        tzina_hron = hron_bill * 0.2901
        chara_hron = hron_bill * 0.2472
        politis_hron = hron_bill * 0.4627

        water_per_apt = water_bill / 3.0 if water_bill > 0 else 0.0

        tzina_total = tzina_hron + 6.70 + 10.33 + water_per_apt
        chara_total = chara_hron + 6.70 + 11.50 + water_per_apt
        politis_total = politis_hron + 6.70 + 11.50 + water_per_apt

        st.success("✅ Ο υπολογισμός ολοκληρώθηκε!")
        st.markdown(f"""
        **Ανακεφαλαίωση {month_input}:**
        * **A1 (Τζίνα):** {tzina_total:.2f} €
        * **B1 (Χαρά):** {chara_total:.2f} €
        * **B2 (Πολίτης):** {politis_total:.2f} €
        """)

# --- ΚΥΡΙΩΣ ΜΕΡΟΣ: CHATBOT UI ---
st.title("🤖 AI Βοηθός Διαχείρισης")
st.caption("Ρώτα οτιδήποτε σχετικά με τον κανονισμό, τα ενοίκια ή τα κοινόχρηστα της πολυκατοικίας.")

# Εμφάνιση Ιστορικού Chat
for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)

# Είσοδος νέας ερώτησης
if user_query := st.chat_input("Γράψτε την ερώτησή σας εδώ..."):
    # 1. Εμφάνιση ερώτησης χρήστη
    st.chat_message("user").markdown(user_query)

    # 2. Απάντηση από το RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Αναζήτηση στα έγγραφα & παραγωγή απάντησης..."):
            response = ask_rag(
                user_query=user_query,
                history=st.session_state.history,
                llm=st.session_state.llm
            )
            st.markdown(response)
            
    # 3. Ενημέρωση ιστορικού
    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.history.append({"role": "assistant", "content": response})