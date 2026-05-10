import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# ==========================================
# CONFIG
# ==========================================
# Lấy Key từ file .env ra
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Kiểm tra xem có lấy được Key không (đề phòng ai đó quên tạo file .env)
if not GEMINI_API_KEY:
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('intfloat/multilingual-e5-base')

model = load_embedding_model()

# ==========================================
# LOAD CHROMADB
# ==========================================
@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path="./db_khoa_cntt")
    return client.get_collection("tai_lieu_khoa_cntt")

collection = load_db()

# ==========================================
# GEMINI
# ==========================================
llm = genai.GenerativeModel("gemini-2.0-flash")

# ==========================================
# UI
# ==========================================
st.title("HCMUTE IT RAG Chatbot")

query = st.text_input("Nhập câu hỏi")

if st.button("Hỏi"):

    # ======================================
    # EMBED QUERY
    # ======================================
    formatted_query = "query: " + query

    query_embedding = model.encode(formatted_query).tolist()

    # ======================================
    # RETRIEVE
    # ======================================
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "distances"]
    )

    documents = results['documents'][0]
    distances = results['distances'][0]

    # ======================================
    # CONTEXT
    # ======================================
    context = "\n\n".join(documents)

    # ======================================
    # PROMPT
    # ======================================
    prompt = f"""
    Bạn là trợ lý tư vấn của Khoa CNTT HCMUTE.

    Chỉ trả lời dựa trên context được cung cấp.
    Nếu không có thông tin thì nói không tìm thấy.

    Context:
    {context}

    Question:
    {query}
    """

    # ======================================
    # GENERATE
    # ======================================
    response = llm.generate_content(prompt)

    # ======================================
    # SHOW ANSWER
    # ======================================
    st.subheader("Câu trả lời")
    st.write(response.text)

    # ======================================
    # SHOW RETRIEVAL
    # ======================================
    st.subheader("Top Retrieved Chunks")

    for i, (doc, distance) in enumerate(zip(documents, distances), 1):

        similarity = 1 - distance

        with st.expander(f"Rank {i} - Similarity {similarity:.4f}"):

            st.write(doc)