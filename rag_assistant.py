import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np


# Load models (embedding and generation)
def load_models():
    embedding_model = SentenceTransformer(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
        cache_folder="./HF-CACHE"
    )
    
    device_id = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3.5-0.8B",
        model_kwargs={"cache_dir": "./HF-CACHE", "torch_dtype": "auto"},
        device=device_id
    )
    return embedding_model, pipe


# Prepare Knowledge Base
def prepare_knowledge(text, embedding_model):
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    embeddings = embedding_model.encode(sentences)
    return sentences, embeddings


# RAG Function
def ask_rag(user_query, history, embedding_model, pipe, knowledge_sentences, knowledge_embeddings, max_history_turns=3):
    # A. Retrieval
    query_embedding = embedding_model.encode([user_query])
    similarities = cosine_similarity(query_embedding, knowledge_embeddings).flatten()
    top_k = 4
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    context = "\n".join([knowledge_chunks[i] for i in top_indices])
    
    
    retrieved_sentences = [knowledge_sentences[i] for i in top_indices]
    context = "\n".join([f"- {s}" for s in retrieved_sentences])

    # B. Sliding Window στο καθαρό ιστορικό (χωρίς τα RAG contexts των προηγούμενων ερωτήσεων)
    window_size = max_history_turns * 2
    if len(history) > window_size + 1:
        active_messages = [history[0]] + history[-window_size:]
    else:
        active_messages = [msg.copy() for msg in history]

    # C. Προσθήκη του Context ΜΟΝΟ στο τελευταίο μήνυμα που θα σταλεί στο LLM
    formatted_user_prompt = f"Information:\n{context}\n\nQuestion: {user_query}"
    active_messages.append({"role": "user", "content": formatted_user_prompt})

    # D. Generation
    output = pipe(
        active_messages, 
        max_new_tokens=150, 
        do_sample=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False  # Επιστρέφει μόνο το νέο κείμενο
    )

    # E. Extract Response
    # Ανάλογα με την έκδοση του Transformers, το output μπορεί να είναι λίστα μηνυμάτων ή dict
    raw_response = output[0]["generated_text"]
    if isinstance(raw_response, list):
        response_text = raw_response[-1]["content"].strip()
    else:
        response_text = raw_response.strip()

    # F. Ενημέρωση του καθολικού ιστορικού με ΚΑΘΑΡΟ κείμενο (για το UI)
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response_text})
    
    return response_text