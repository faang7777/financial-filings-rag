import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# ---------- Page setup ----------
st.set_page_config(
    page_title="Financial Filings Q&A",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Filings Q&A Assistant")
st.caption("Ask questions about Apple, Microsoft, and Tesla 10-K filings")

# ---------- Cached resource loading ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    """Load pre-computed chunks and embeddings."""
    with open('chunks.json', 'r') as f:
        chunks = json.load(f)
    
    embeddings = np.load('embeddings.npy')
    
    # Normalize for cosine similarity (faster lookups later)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms
    
    return chunks, embeddings_normalized

@st.cache_resource
def load_gemini():
    api_key = st.secrets["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)

# ---------- Helper functions ----------
def detect_company(question):
    question_lower = question.lower()
    keywords = {
        "AAPL": ["apple", "iphone", "ipad", "mac"],
        "MSFT": ["microsoft", "azure", "windows", "xbox"],
        "TSLA": ["tesla", "musk", "model 3", "model y", "cybertruck"]
    }
    return [t for t, kws in keywords.items() if any(kw in question_lower for kw in kws)]

def search_chunks(question, n_chunks=8):
    """Find most relevant chunks using cosine similarity."""
    embedder = load_embedder()
    chunks, embeddings = load_data()
    
    # Embed the question and normalize
    q_emb = embedder.encode([question])[0]
    q_emb = q_emb / np.linalg.norm(q_emb)
    
    # Detect company filter
    detected = detect_company(question)
    
    if detected:
        # Filter by ticker first
        mask = np.array([c["ticker"] in detected for c in chunks])
        valid_indices = np.where(mask)[0]
        valid_embeddings = embeddings[valid_indices]
        
        # Cosine similarity (dot product since normalized)
        scores = valid_embeddings @ q_emb
        top_local = np.argsort(scores)[-n_chunks:][::-1]
        top_indices = valid_indices[top_local]
    else:
        scores = embeddings @ q_emb
        top_indices = np.argsort(scores)[-n_chunks:][::-1]
    
    return [chunks[i] for i in top_indices]

def ask_question(question, n_chunks=8):
    client = load_gemini()
    relevant = search_chunks(question, n_chunks)
    
    context = "\n\n---\n\n".join([
        f"[Source: {c['ticker']} 10-K]\n{c['text']}" for c in relevant
    ])
    
    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY the context below from SEC 10-K filings.

If the context doesn't contain enough information, say so honestly. Always cite which company's filing you're drawing from. Be specific — include numbers, dates, and direct details when present.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text, relevant

# ---------- UI ----------
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses **Retrieval-Augmented Generation (RAG)** to answer questions about SEC 10-K filings.
    
    **Companies covered:**
    - 🍎 Apple (AAPL)
    - 🪟 Microsoft (MSFT)
    - 🚗 Tesla (TSLA)
    
    **Tech stack:**
    - Embeddings: sentence-transformers (MiniLM-L6-v2)
    - Vector search: NumPy cosine similarity
    - LLM: Gemini 2.5 Flash-Lite
    - Source: SEC EDGAR
    """)

st.subheader("Try a sample question:")
samples = [
    "What were Apple's total revenues in the most recent fiscal year?",
    "How many employees does Tesla have?",
    "What competitive risks does Microsoft face in cloud computing?",
    "What are the main risk factors Tesla identifies?"
]

cols = st.columns(2)
for i, sample in enumerate(samples):
    if cols[i % 2].button(sample, key=f"sample_{i}", use_container_width=True):
        st.session_state['question'] = sample

question = st.text_input(
    "Or ask your own question:",
    value=st.session_state.get('question', ''),
    placeholder="e.g., How much did Microsoft spend on R&D?"
)

if st.button("Get Answer", type="primary") and question:
    with st.spinner("Searching filings and generating answer..."):
        try:
            answer, sources = ask_question(question)
            
            st.subheader("Answer")
            st.write(answer)
            
            with st.expander("📚 Show source chunks used"):
                for i, c in enumerate(sources):
                    st.markdown(f"**Chunk {i+1} — {c['ticker']} 10-K**")
                    st.text(c['text'][:500] + "..." if len(c['text']) > 500 else c['text'])
                    st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
