"""
RAG Chatbox - Streamlit App
D:\simpleRAGchatbox\
    chatbox.py
    .env                ← GROQ_API_KEY=gsk_...
    paper\*.txt
    index\              ← tự tạo khi chạy lần đầu
"""

import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ──────────────────────────────────────────────
# 1. Load và kiểm tra API key
# ──────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    st.error("❌ Không tìm thấy GROQ_API_KEY trong file .env")
    st.info("Thêm vào file .env:\nGROQ_API_KEY=gsk_xxxxxxxxxxxx")
    st.stop()

if not GROQ_API_KEY.startswith("gsk_"):
    st.error(f"❌ API key phải bắt đầu bằng 'gsk_', hiện tại: '{GROQ_API_KEY[:6]}...'")
    st.stop()

try:
    _test = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0)
    _test.invoke("hi")
except Exception as e:
    err = str(e)
    if "401" in err or "invalid_api_key" in err:
        st.error("❌ API key Groq không hợp lệ → https://console.groq.com/keys")
    elif "429" in err:
        st.error("❌ Groq hết quota hôm nay, thử lại sau.")
    else:
        st.error(f"❌ Lỗi kết nối Groq: {err}")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ──────────────────────────────────────────────
# 2. Đường dẫn
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(BASE_DIR, "paper")
INDEX_DIR = os.path.join(BASE_DIR, "index")

# ──────────────────────────────────────────────
# FIX 2: Tách mỗi section thành 2 chunk riêng
#   Chunk A: [HEADER] + nội dung chính + ý nghĩa nội hàm
#   Chunk B: [HEADER] + biến thể câu hỏi (dòng "Các biến thể...")
# Chunk B match câu hỏi user → dẫn về đúng section
# ──────────────────────────────────────────────
def split_section_into_chunks(header: str, body: str, source: str) -> list[Document]:
    chunks = []

    # Tách phần biến thể câu hỏi ra khỏi body
    # Pattern: dòng bắt đầu bằng "- Các biến thể câu hỏi"
    variant_pattern = r'(-\s*Các biến thể câu hỏi[^\n]*\n)(.*)'
    match = re.search(variant_pattern, body, re.DOTALL)

    if match:
        main_body    = body[:match.start()].strip()
        variant_body = match.group(0).strip()

        # Chunk A: tiêu đề + nội dung chính
        chunk_a = f"{header}\n{main_body}".strip()
        if chunk_a:
            chunks.append(Document(
                page_content=chunk_a,
                metadata={"source": source, "section": header, "chunk_type": "content"}
            ))

        # Chunk B: tiêu đề + biến thể câu hỏi
        chunk_b = f"{header}\n{variant_body}".strip()
        if chunk_b:
            chunks.append(Document(
                page_content=chunk_b,
                metadata={"source": source, "section": header, "chunk_type": "variants"}
            ))
    else:
        # Không có biến thể → giữ nguyên 1 chunk
        full = f"{header}\n{body}".strip()
        if full:
            chunks.append(Document(
                page_content=full,
                metadata={"source": source, "section": header, "chunk_type": "content"}
            ))

    return chunks


def parse_documents(raw_docs: list) -> list[Document]:
    all_chunks = []
    for doc in raw_docs:
        source = doc.metadata.get("source", "unknown")
        text   = doc.page_content

        # Tách theo [HEADER SECTION]
        pattern = r'(\[[^\]]+\])'
        parts   = re.split(pattern, text)

        # Text trước section đầu tiên (intro)
        if parts and not parts[0].strip().startswith('['):
            intro = parts[0].strip()
            if intro:
                all_chunks.append(Document(
                    page_content=intro,
                    metadata={"source": source, "section": "intro", "chunk_type": "content"}
                ))
            parts = parts[1:]

        # Duyệt từng cặp [HEADER] + body
        i = 0
        while i < len(parts) - 1:
            header = parts[i].strip()
            body   = parts[i+1].strip() if i+1 < len(parts) else ""
            if header.startswith('[') and body:
                chunks = split_section_into_chunks(header, body, source)
                all_chunks.extend(chunks)
            i += 2

        # Fallback nếu không parse được gì
        if not all_chunks:
            all_chunks.append(Document(
                page_content=text,
                metadata={"source": source, "section": "all", "chunk_type": "content"}
            ))

    return all_chunks

# ──────────────────────────────────────────────
# FIX 1: Đổi embedding sang multilingual-e5-small
# Được train đặc biệt cho retrieval, hiểu ngữ nghĩa
# tiếng Việt tốt hơn MiniLM nhiều
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải embedding model (lần đầu ~2 phút)...")
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(PAPER_DIR):
        st.error(f"❌ Không tìm thấy thư mục paper: {PAPER_DIR}")
        st.stop()

    loader = DirectoryLoader(
        PAPER_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    raw_docs = loader.load()
    if not raw_docs:
        st.error(f"❌ Không có file .txt nào trong: {PAPER_DIR}")
        st.stop()

    all_chunks = parse_documents(raw_docs)
    vectordb   = FAISS.from_documents(all_chunks, embeddings)
    vectordb.save_local(INDEX_DIR)
    return vectordb

# ──────────────────────────────────────────────
# 4. LLM + Retriever
# Khi retriever tìm chunk B (variants) → trả về section
# → LLM đọc chunk A (content) cùng section để trả lời
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Khởi tạo AI...")
def load_llm_retriever(_vectordb):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )
    retriever = _vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}   # lấy 6 vì mỗi section có 2 chunk (A+B)
    )
    return llm, retriever


def dedupe_by_section(docs: list[Document]) -> list[Document]:
    """Gộp chunk A và B cùng section lại, ưu tiên giữ chunk content."""
    seen_sections = {}
    for doc in docs:
        section = doc.metadata.get("section", "")
        ctype   = doc.metadata.get("chunk_type", "content")
        if section not in seen_sections:
            seen_sections[section] = doc
        elif ctype == "content":
            # Ưu tiên chunk content hơn chunk variants
            seen_sections[section] = doc
    return list(seen_sections.values())

# ──────────────────────────────────────────────
# 5. Hàm trả lời
# ──────────────────────────────────────────────
def get_answer(llm, retriever, user_input, history):
    raw_docs = retriever.invoke(user_input)
    docs     = dedupe_by_section(raw_docs)   # gộp A+B → chỉ giữ nội dung

    context_parts = []
    for i, d in enumerate(docs):
        src     = os.path.basename(d.metadata.get("source", "unknown"))
        section = d.metadata.get("section", "")
        context_parts.append(f"[Đoạn {i+1} | {src} | {section}]\n{d.page_content}")
    context = "\n\n".join(context_parts)

    history_text = ""
    for m in history:
        role = "Người dùng" if m["role"] == "user" else "Trợ lý"
        history_text += f"{role}: {m['content']}\n"

    prompt = ChatPromptTemplate.from_template(
        "Bạn là trợ lý AI của tổ chức, chuyên trả lời câu hỏi dựa trên tài liệu nội bộ.\n\n"
        "NGUYÊN TẮC:\n"
        "1. Chỉ dùng thông tin trong 'NGỮ CẢNH TÀI LIỆU' bên dưới.\n"
        "2. Nếu không tìm thấy thông tin → nói rõ: 'Tôi không tìm thấy thông tin này trong tài liệu.'\n"
        "3. Trả lời ngắn gọn, rõ ràng, đúng trọng tâm câu hỏi.\n"
        "4. Nếu câu hỏi liên quan đến quy định/điều cấm → trích dẫn rõ nội dung quy định.\n\n"
        "NGỮ CẢNH TÀI LIỆU:\n{context}\n\n"
        "LỊCH SỬ HỘI THOẠI:\n{history}\n\n"
        "CÂU HỎI: {question}\n\n"
        "TRẢ LỜI:"
    )

    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context":  context,
        "history":  history_text,
        "question": user_input,
    })
    return answer, docs

# ──────────────────────────────────────────────
# 6. Streamlit UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbox", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbox")
st.caption("Hỏi đáp thông tin từ tài liệu nội bộ")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("🔑 Nhập mật khẩu nội bộ:", type="password")
    if pwd == "123456":
        st.session_state.authenticated = True
        st.rerun()
    elif pwd:
        st.error("❌ Sai mật khẩu.")
    st.stop()

vectordb       = load_vectordb()
llm, retriever = load_llm_retriever(vectordb)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Cài đặt")
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**LLM:** Llama 3.3 70B (Groq)")
    st.markdown("**Embedding:** multilingual-e5-small")
    st.markdown("**Chunking:** section + variants split")
    st.success("✅ Sẵn sàng")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            answer, sources = get_answer(llm, retriever, user_input, history)

        st.markdown(answer)

        if sources:
            with st.expander("📄 Nguồn tham khảo"):
                seen = set()
                for doc in sources:
                    src     = os.path.basename(doc.metadata.get("source", "unknown"))
                    section = doc.metadata.get("section", "")
                    key     = f"{src}::{section}"
                    if key not in seen:
                        st.markdown(f"- `{src}` — {section}")
                        seen.add(key)

    st.session_state.messages.append({"role": "assistant", "content": answer})
