import streamlit as st
import json
import chromadb
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

# ---------- Load resources (cached so this runs only once) ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_vector_db():
    """Load chunks from JSON and build ChromaDB collection."""
    with open('chunks.json', 'r') as f:
        all_chunks = json.load(f)
    
    embedder = load_embedder()
    
    # Build vector DB in memory
    client = chromadb.Client()
    try:
        client.delete_collection("filings")
    except:
        pass
    collection = client.create_collection(name="filings")
    
    # Embed and add in batches
    texts = [c["text"] for c in all_chunks]
    
    progress_bar = st.progress(0, text="Building vector database (one-time setup)...")
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=False)
    progress_bar.progress(50, text="Storing vectors...")
    
    batch_size = 5000
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            embeddings=[e.tolist() for e in embeddings[i:end]],
            documents=texts[i:end],
            metadatas=[{"ticker": c["ticker"]} for c in all_chunks[i:end]],
            ids=[c["chunk_id"] for c in all_chunks[i:end]]
        )
    
    progress_bar.empty()
    return collection

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

def ask_question(question, n_chunks=8):
    embedder = load_embedder()
    collection = load_vector_db()
    client = load_gemini()
    
    detected = detect_company(question)
    q_embedding = embedder.encode([question])[0].tolist()
    
    if detected:
        where = {"ticker": {"$in": detected}} if len(detected) > 1 else {"ticker": detected[0]}
        results = collection.query(query_embeddings=[q_embedding], n_results=n_chunks, where=where)
    else:
        results = collection.query(query_embeddings=[q_embedding], n_results=n_chunks)
    
    chunks = results['documents'][0]
    metas = results['metadatas'][0]
    
    context = "\n\n---\n\n".join([
        f"[Source: {m['ticker']} 10-K]\n{c}" for c, m in zip(chunks, metas)
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
    return response.text, chunks, metas

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
    - Embeddings: sentence-transformers
    - Vector DB: ChromaDB
    - LLM: Gemini 2.5 Flash-Lite
    - Source: SEC EDGAR
    """)

# Sample questions
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

# Question input
question = st.text_input(
    "Or ask your own question:",
    value=st.session_state.get('question', ''),
    placeholder="e.g., How much did Microsoft spend on R&D?"
)

if st.button("Get Answer", type="primary") and question:
    with st.spinner("Retrieving relevant filings and generating answer..."):
        try:
            answer, chunks, metas = ask_question(question)
            
            st.subheader("Answer")
            st.write(answer)
            
            with st.expander("📚 Show source chunks used"):
                for i, (c, m) in enumerate(zip(chunks, metas)):
                    st.markdown(f"**Chunk {i+1} — {m['ticker']} 10-K**")
                    st.text(c[:500] + "..." if len(c) > 500 else c)
                    st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
