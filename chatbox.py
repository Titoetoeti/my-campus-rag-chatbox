"""
RAG Chatbox - Streamlit App (Optimized version)
D:\simpleRAGchatbox\
    chatbox.py
    .env                ← chứa GROQ_API_KEY=gsk_...
    paper\*.txt
    index\              ← tự tạo khi chạy lần đầu
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ──────────────────────────────────────────────
# 1. Load và KIỂM TRA API key ngay từ đầu
# ──────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    st.error("❌ Không tìm thấy GROQ_API_KEY trong file .env")
    st.info("Tạo file .env trong thư mục D:\\simpleRAGchatbox\\ với nội dung:\nGROQ_API_KEY=gsk_xxxxxxxxxxxx")
    st.stop()

if not GROQ_API_KEY.startswith("gsk_"):
    st.error(f"❌ API key không hợp lệ. Key phải bắt đầu bằng 'gsk_', key hiện tại bắt đầu bằng: '{GROQ_API_KEY[:6]}...'")
    st.info("Vào https://console.groq.com/keys để lấy key đúng")
    st.stop()

# Thử kết nối Groq ngay khi khởi động để báo lỗi sớm
try:
    _test_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)
    _test_llm.invoke("hi")
except Exception as e:
    err = str(e)
    if "401" in err or "invalid_api_key" in err:
        st.error("❌ API key Groq không hợp lệ (lỗi 401)")
        st.info("Vào https://console.groq.com/keys → tạo key mới → paste vào file .env")
    elif "429" in err:
        st.error("❌ Groq API đã hết quota hôm nay (lỗi 429). Thử lại sau.")
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
# 3. Vector DB — Tối ưu hóa Chunking chiến lược
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải embedding model (lần đầu ~1 phút)...")
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(PAPER_DIR):
        st.error(f"❌ Không tìm thấy thư mục paper tại: {PAPER_DIR}")
        st.stop()

    loader = DirectoryLoader(
        PAPER_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    if not docs:
        st.error(f"❌ Không tìm thấy file .txt nào trong: {PAPER_DIR}")
        st.stop()

    # CẢI TIẾN 1: Tách text thông minh theo dấu câu, tăng độ dài chunk hợp lý với mô hình multilingual
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    ).split_documents(docs)
    
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(INDEX_DIR)
    return vectordb

# ──────────────────────────────────────────────
# 4. LLM + Retriever (Kích hoạt thuật toán MMR chống trùng lặp văn bản)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Khởi tạo AI...")
def load_llm_retriever(_vectordb):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.2, # Giảm nhẹ nhiệt độ để LLM bớt "sáng tạo" linh tinh, tập trung vào tài liệu
    )
    
    # CẢI TIẾN 2: Sử dụng MMR (Maximal Marginal Relevance) để tối ưu việc tìm kiếm
    # Giúp lấy ra các đoạn văn bản vừa liên quan nhất, vừa KHÁC BIỆT NHAU nhất (tránh lấy trùng ý), nâng k lên 4-5 để giàu ngữ cảnh hơn.
    retriever = _vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
    )
    return llm, retriever

# ──────────────────────────────────────────────
# 5. Hàm trả lời — Giao thức Tối ưu câu hỏi (Query Condensation) + Bọc thẻ Ngữ cảnh
# ──────────────────────────────────────────────
def get_answer(llm, retriever, user_input, history):
    # 1. Định dạng lịch sử trò chuyện dạng chuỗi sạch
    history_text = ""
    for m in history:
        role = "Người dùng" if m["role"] == "user" else "Trợ lý"
        history_text += f"{role}: {m['content']}\n"

    # 2. THUẬT TOÁN MỞ RỘNG TRUY VẤN (Query Expansion & Synonym Mapping)
    # Ép LLM sinh ra câu truy vấn chứa tất cả các từ đồng nghĩa hành chính để Vector DB quét không trượt phát nào.
    expansion_prompt = (
        "Bạn là chuyên gia tối ưu hóa truy vấn cho hệ thống tìm kiếm tài liệu nội bộ.\n"
        "Nhiệm vụ của bạn là đọc câu hỏi của người dùng (và lịch sử nếu có), sau đó viết lại thành một câu "
        "truy vấn mới chứa đầy đủ các TỪ ĐỒNG NGHĨA và THUẬT NGỮ thường dùng trong văn bản quy định.\n"
        "Quy tắc chuyển đổi bắt buộc:\n"
        "- Nếu user hỏi về 'không được làm', 'không được phép làm', 'cấm làm' -> Phải mở rộng thành 'hành vi bị cấm, quy định nghiêm cấm, không được phép'.\n"
        "- Nếu user hỏi về 'đi muộn', 'vắng mặt' -> Mở rộng thành 'vi phạm giờ giấc, kỷ luật lao động'.\n"
        "- Nếu user hỏi về 'tiền nong', 'thưởng' -> Mở rộng thành 'chế độ đãi ngộ, lương thưởng'.\n\n"
        f"Lịch sử hội thoại:\n{history_text}\n"
        f"Câu hỏi của người dùng: {user_input}\n"
        "Câu truy vấn tối ưu chứa từ đồng nghĩa (Chỉ trả về câu văn kết quả, TUYỆT ĐỐI không giải thích gì thêm):"
    )
    
    search_query = user_input
    try:
        expanded_res = llm.invoke(expansion_prompt).content.strip()
        if expanded_res:
            search_query = expanded_res
    except Exception:
        pass # Dự phòng nếu lỗi API Groq thì dùng câu gốc của user

    # 3. Tiến hành tìm kiếm bằng câu truy vấn đã được "bơm" từ đồng nghĩa
    docs = retriever.invoke(search_query)
    
    # 4. Bọc cấu trúc tường minh kèm tên File nguồn cho từng đoạn ngữ cảnh
    context_parts = []
    for i, d in enumerate(docs):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        context_parts.append(f"[Đoạn {i+1} - Nguồn: {src}]:\n{d.page_content}")
    context = "\n\n".join(context_parts)

    # 5. Prompt trả lời nghiêm ngặt
    prompt = ChatPromptTemplate.from_template(
        "Bạn là trợ lý AI chuyên nghiệp của tổ chức, hỗ trợ người dùng tìm kiếm thông tin.\n"
        "Nhiệm vụ của bạn là trả lời câu hỏi hiện tại một cách chính xác, trung thực dựa TRÊN NGỮ CẢNH TÀI LIỆU được cung cấp.\n\n"
        "--- NGUYÊN TẮC QUAN TRỌNG ---\n"
        "1. Chỉ sử dụng thông tin có trong phần 'Ngữ cảnh tài liệu'. Không tự ý suy diễn hoặc dùng kiến thức bên ngoài.\n"
        "2. Nếu tài liệu không chứa câu trả lời, hãy nói: 'Tôi không tìm thấy thông tin này trong tài liệu nội bộ.' TUYỆT ĐỐI không được bịa đặt câu trả lời.\n"
        "3. Trả lời ngắn gọn, trực diện vào câu hỏi.\n\n"
        "--- NGỮ CẢNH TÀI LIỆU ---\n{context}\n\n"
        "--- LỊCH SỬ HỘI THOẠI ---\n{history}\n\n"
        "--- CÂU HỎI HIỆN TẠI ---\n{question}\n\n"
        "Câu trả lời của bạn:"
    )
    
    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "history": history_text, "question": user_input})
    return answer, docs

# ──────────────────────────────────────────────
# 6. Streamlit UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbox", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbox")
st.caption("Hỏi đáp thông tin từ tài liệu nội bộ (giới thiệu, nội quy, tầm nhìn...)")

vectordb       = load_vectordb()
llm, retriever = load_llm_retriever(vectordb)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Cài đặt")
    if st.button("🗑️ Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**LLM:** Llama 3.1 8B (Groq)")
    st.markdown("**Embedding:** multilingual-MiniLM (local)")
    st.markdown("**Vector DB:** FAISS (local)")
    st.success("✅ API key hợp lệ")

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
                    src = os.path.basename(doc.metadata.get("source", "unknown"))
                    if src not in seen:
                        st.markdown(f"- `{src}`")
                        seen.add(src)

    st.session_state.messages.append({"role": "assistant", "content": answer})