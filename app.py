import streamlit as st
from sentence_transformers import SentenceTransformer
import traceback

st.set_page_config(page_title="HCMUTE IT RAG Chatbot", layout="wide")

DB_PATH = "./db_khoa_cntt"
COLLECTION_NAME = "tai_lieu_khoa_cntt"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
OLLAMA_MODEL = "phi3.5"
TOP_K = 5

st.title("HCMUTE IT RAG Chatbot")
st.write("Hỏi đáp về thông tin của Khoa Công nghệ Thông tin - HCMUTE")
st.markdown("---")


@st.cache_resource
def load_embedding_model():
    st.write("STEP 1: Loading embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_collection():
    st.write("STEP 2: Loading ChromaDB...")
    import chromadb
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION_NAME)


def ask_ollama(prompt):
    st.write("STEP 5: Calling Ollama...")
    import ollama
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


query = st.text_input(
    "Nhập câu hỏi của bạn:",
    placeholder="Ví dụ: Khoa CNTT có những bộ môn nào?"
)

if st.button("Hỏi"):
    if not query.strip():
        st.warning("Vui lòng nhập câu hỏi!")
        st.stop()

    try:
        model = load_embedding_model()
        collection = load_collection()

        st.write("STEP 3: Encoding query...")
        query_embedding = model.encode(
            "query: " + query,
            normalize_embeddings=True
        ).tolist()

        st.write("STEP 4: Querying ChromaDB...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "distances", "metadatas"]
        )

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        context_blocks = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            source = meta.get("source_url", "Không rõ nguồn")
            context_blocks.append(
                f"[Tài liệu {i}]\nNguồn: {source}\nNội dung:\n{doc}"
            )

        context = "\n\n".join(context_blocks)

        prompt = f"""
Bạn là chatbot tư vấn của Khoa Công nghệ Thông tin HCMUTE.

Quy tắc trả lời:
- Chỉ trả lời dựa trên Context.
- Trả lời đúng trọng tâm câu hỏi.
- Không lan man sang thông tin không được hỏi.
- Nếu Context không có thông tin phù hợp, hãy trả lời: "Dữ liệu hiện tại chưa có thông tin này."
- Không tự bịa thông tin ngoài Context.

Context:
{context}

Câu hỏi:
{query}

Câu trả lời:
"""

        answer = ask_ollama(prompt)

        st.subheader("Câu trả lời")
        st.write(answer)

        st.subheader("Top Retrieved Chunks")
        for i, (doc, distance, meta) in enumerate(zip(documents, distances, metadatas), 1):
            similarity = 1 - distance
            source = meta.get("source_url", "Không rõ nguồn")

            with st.expander(f"Rank {i} - Similarity {similarity:.4f}"):
                st.write(f"Nguồn: {source}")
                st.write(doc)

    except Exception as e:
        st.error("Có lỗi xảy ra:")
        st.code(traceback.format_exc())