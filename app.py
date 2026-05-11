import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ==========================================
# PAGE CONFIG (MUST BE FIRST)
# ==========================================
st.set_page_config(page_title="HCMUTE IT RAG Chatbot", layout="wide")

# ==========================================
# CONFIG
# ==========================================
# Lấy Key từ file .env ra
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Kiểm tra xem có lấy được Key không (đề phòng ai đó quên tạo file .env)
if not GEMINI_API_KEY:
    st.error("Lỗi: Chưa set biến môi trường GEMINI_API_KEY. Vui lòng kiểm tra file .env")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Lỗi khi configure Gemini: {str(e)}")
    st.stop()

# ==========================================
# LIST AVAILABLE MODELS
# ==========================================
selected_model = None
try:
    available_models = [model.name for model in genai.list_models() if 'generateContent' in model.supported_generation_methods]
    
    # Select the first available model (silent mode, no UI)
    if available_models:
        selected_model = available_models[0].replace('models/', '')
    else:
        st.error("❌ Không có model nào khả dụng!")
        st.stop()
except Exception as e:
    st.error(f"Lỗi khi liệt kê model: {str(e)}")
    st.stop()

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('intfloat/multilingual-e5-base')
    except Exception as e:
        st.error(f"Lỗi khi load embedding model: {str(e)}")
        st.stop()

try:
    model = load_embedding_model()
except Exception as e:
    st.error(f"Lỗi: {str(e)}")
    st.stop()

# ==========================================
# LOAD CHROMADB
# ==========================================
@st.cache_resource
def load_db():
    try:
        client = chromadb.PersistentClient(path="./db_khoa_cntt")
        collection = client.get_collection("tai_lieu_khoa_cntt")
        return collection
    except Exception as e:
        st.error(f"Lỗi khi load ChromaDB: {str(e)}")
        st.stop()

try:
    collection = load_db()
except Exception as e:
    st.error(f"Lỗi: {str(e)}")
    st.stop()

# ==========================================
# GEMINI
# ==========================================
llm = genai.GenerativeModel(selected_model)

# ==========================================
# UI
# ==========================================
st.title(" HCMUTE IT RAG Chatbot")
st.write("Hỏi đáp về các thông tin của Khoa CNTT - Trường ĐH Công nghệ Kỹ thuật TP.HCM")

st.markdown("---")

query = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Khoa CNTT có những ngành học nào?")

if st.button("Hỏi"):
    if not query.strip():
        st.warning("⚠️ Vui lòng nhập câu hỏi!")
    else:
        try:
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
        except Exception as e:
            st.error(f" Lỗi khi xử lý câu hỏi: {str(e)}")