r"""
RAG Chatbox - Streamlit App (PDF-native, 5-stage pipeline)
D:\simpleRAGchatbox\
    chatbox.py
    rag_pipeline.py       <- nhạc trưởng, ghép 5 giai đoạn
    query_transform.py    <- Giai đoạn 1: Query Transformation
    hybrid_retriever.py   <- Giai đoạn 2: Hybrid Retrieval (Vector + BM25 + RRF)
    reranker.py           <- Giai đoạn 3: Reranking (BGE Cross-Encoder local)
    corrective_rag.py     <- Giai đoạn 4: Corrective RAG (chỉ chấm điểm, không web search)
    generation.py         <- Giai đoạn 5: Generation + Guard (Llama Guard)
    chat_logger.py        <- Ghi log hỏi-đáp ra CSV (xem lại bằng Excel) cho mục đích quản lý
    .env                  <- GROQ_API_KEY=gsk_...
    paper\*.pdf           <- DOC TRUC TIEP PDF
    index\                <- tu tao khi chay lan dau
    logs\chat_log.csv     <- tu tao khi co luot hoi dau tien
"""

import os
import re
import shutil
from datetime import date
import streamlit as st
from dotenv import load_dotenv

import pdfplumber
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from rag_pipeline import answer_question
from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker
from chat_logger import log_turn, LOG_FILE

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
# 3. Đọc và chunk PDF theo cấu trúc văn bản pháp lý
#
# Chiến lược chunking (giữ nguyên như bản gốc — đã hoạt động tốt):
#   - Tách theo Điều (Điều 1., Điều 2., ...) — đơn vị tri thức nhỏ nhất
#   - Mỗi chunk giữ nguyên tiêu đề Chương cha để tăng ngữ cảnh
#   - Chunk phụ lục (bảng vi phạm) tách riêng theo số thứ tự hàng
#   - Khi chunk quá dài (>1200 ký tự) → tách theo khoảng trắng đoạn
# ──────────────────────────────────────────────

RE_CHUONG   = re.compile(r'^(Chương\s+[IVXLCDM]+\b.*)', re.MULTILINE)
RE_DIEU     = re.compile(r'^(\s*Điều\s+\d+[\.\:][^\n]*)', re.MULTILINE)
RE_PHAN     = re.compile(r'^(Phụ\s+lục\s+[IVXLCDM\d]+\b.*)', re.MULTILINE | re.IGNORECASE)
MAX_CHUNK   = 1200   # ký tự tối đa mỗi chunk


def _split_long(text: str, max_len: int = MAX_CHUNK) -> list[str]:
    """Chia đoạn văn dài thành các chunk nhỏ hơn theo dòng trống."""
    if len(text) <= max_len:
        return [text]
    parts = re.split(r'\n{2,}', text)
    chunks, buf = [], ""
    for p in parts:
        if len(buf) + len(p) + 2 > max_len and buf:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = (buf + "\n\n" + p).strip() if buf else p
    if buf:
        chunks.append(buf.strip())
    return chunks if chunks else [text[:max_len]]


def extract_text_from_pdf(pdf_path: str) -> str:
    """Trích xuất toàn bộ text từ PDF bằng pdfplumber."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t.strip())
    return "\n\n".join(pages_text)


def chunk_legal_text(full_text: str, source: str) -> list[Document]:
    """
    Tách văn bản pháp lý thành các Document theo cấu trúc:
      Chương → Điều → khoản/nội dung

    Mỗi Document mang metadata:
      source, chapter (Chương cha), section (tiêu đề Điều), page_hint
    """
    docs = []
    current_chapter = ""

    split_points = []
    for m in RE_CHUONG.finditer(full_text):
        split_points.append(('chapter', m.start(), m.group(1)))
    for m in RE_PHAN.finditer(full_text):
        split_points.append(('annex', m.start(), m.group(1)))
    split_points.sort(key=lambda x: x[1])

    if split_points:
        intro_text = full_text[:split_points[0][1]].strip()
    else:
        intro_text = full_text.strip()

    if intro_text:
        for chunk in _split_long(intro_text):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": source, "chapter": "Giới thiệu", "section": "intro", "chunk_type": "content"}
            ))

    for idx, (kind, start, title) in enumerate(split_points):
        end = split_points[idx + 1][1] if idx + 1 < len(split_points) else len(full_text)
        block = full_text[start:end].strip()

        if kind == 'chapter':
            current_chapter = title.strip()
            _chunk_chapter(block, current_chapter, source, docs)

        elif kind == 'annex':
            for chunk in _split_long(block):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": source, "chapter": title.strip(), "section": title.strip(), "chunk_type": "annex"}
                ))

    return docs


def _chunk_chapter(chapter_text: str, chapter_name: str, source: str, docs: list):
    """Chia một Chương thành các chunk theo Điều."""
    dieu_matches = list(RE_DIEU.finditer(chapter_text))

    if not dieu_matches:
        for chunk in _split_long(chapter_text):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": source, "chapter": chapter_name, "section": chapter_name, "chunk_type": "content"}
            ))
        return

    preamble = chapter_text[:dieu_matches[0].start()].strip()
    if preamble:
        docs.append(Document(
            page_content=f"{chapter_name}\n{preamble}",
            metadata={"source": source, "chapter": chapter_name, "section": chapter_name + " - Mở đầu", "chunk_type": "content"}
        ))

    for i, m in enumerate(dieu_matches):
        dieu_title = m.group(1).strip()
        dieu_start = m.start()
        dieu_end   = dieu_matches[i + 1].start() if i + 1 < len(dieu_matches) else len(chapter_text)
        dieu_body  = chapter_text[dieu_start:dieu_end].strip()

        header = f"{chapter_name} > {dieu_title}"
        full   = f"{header}\n{dieu_body}"

        for chunk in _split_long(full):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": source, "chapter": chapter_name, "section": dieu_title, "chunk_type": "content"}
            ))


def load_pdf_documents(paper_dir: str) -> list[Document]:
    """Quét toàn bộ file PDF trong thư mục paper/ và trả về danh sách Document."""
    all_docs = []
    pdf_files = [f for f in os.listdir(paper_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        return all_docs

    for fname in pdf_files:
        fpath = os.path.join(paper_dir, fname)
        try:
            text = extract_text_from_pdf(fpath)
            if not text.strip():
                continue
            chunks = chunk_legal_text(text, fname)
            all_docs.extend(chunks)
        except Exception as e:
            st.warning(f"⚠️ Không đọc được {fname}: {e}")

    return all_docs


# ── Giữ lại parser TXT cũ như fallback (không xóa) ──
def _parse_txt_documents(raw_docs: list) -> list[Document]:
    """Parser TXT cũ với [HEADER] format — dùng khi không có PDF."""
    all_chunks = []
    for doc in raw_docs:
        source = doc.metadata.get("source", "unknown")
        text   = doc.page_content
        pattern = r'(\[[^\]]+\])'
        parts   = re.split(pattern, text)
        if parts and not parts[0].strip().startswith('['):
            intro = parts[0].strip()
            if intro:
                all_chunks.append(Document(page_content=intro, metadata={"source": source, "section": "intro", "chunk_type": "content"}))
            parts = parts[1:]
        i = 0
        while i < len(parts) - 1:
            header = parts[i].strip()
            body   = parts[i+1].strip() if i+1 < len(parts) else ""
            if header.startswith('[') and body:
                variant_pattern = r'(-\s*Các biến thể câu hỏi[^\n]*\n)(.*)'
                m = re.search(variant_pattern, body, re.DOTALL)
                if m:
                    main_body = body[:m.start()].strip()
                    var_body  = m.group(0).strip()
                    if main_body:
                        all_chunks.append(Document(page_content=f"{header}\n{main_body}", metadata={"source": source, "section": header, "chunk_type": "content"}))
                    if var_body:
                        all_chunks.append(Document(page_content=f"{header}\n{var_body}", metadata={"source": source, "section": header, "chunk_type": "variants"}))
                else:
                    full = f"{header}\n{body}".strip()
                    if full:
                        all_chunks.append(Document(page_content=full, metadata={"source": source, "section": header, "chunk_type": "content"}))
            i += 2
        if not all_chunks:
            all_chunks.append(Document(page_content=text, metadata={"source": source, "section": "all", "chunk_type": "content"}))
    return all_chunks


# ──────────────────────────────────────────────
# 4. Vector DB — hỗ trợ cả PDF lẫn TXT (ưu tiên PDF)
#    Trả về cả vectordb (FAISS) lẫn all_chunks (list[Document]) vì
#    Giai đoạn 2 (Hybrid Retrieval) cần all_chunks để khởi tạo BM25
#    (BM25 không "index sẵn" như FAISS — cần load toàn bộ corpus mỗi lần).
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải embedding model (lần đầu ~2 phút)...")
def load_vectordb_and_chunks():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Ưu tiên đọc PDF ──
    all_chunks = load_pdf_documents(PAPER_DIR)

    # ── Fallback: nếu không có PDF nào thì đọc TXT như cũ ──
    if not all_chunks:
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        loader = DirectoryLoader(
            PAPER_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        raw_docs = loader.load()
        if not raw_docs:
            st.error(f"❌ Không có file PDF hoặc TXT nào trong: {PAPER_DIR}")
            st.stop()
        all_chunks = _parse_txt_documents(raw_docs)

    if os.path.exists(INDEX_DIR):
        vectordb = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(PAPER_DIR):
            st.error(f"❌ Không tìm thấy thư mục paper: {PAPER_DIR}")
            st.stop()
        vectordb = FAISS.from_documents(all_chunks, embeddings)
        vectordb.save_local(INDEX_DIR)

    return vectordb, all_chunks


# ──────────────────────────────────────────────
# 5. Khởi tạo pipeline 5 giai đoạn
#    LLM chính (trả lời) + HybridRetriever (Giai đoạn 2) + Reranker (Giai đoạn 3).
#    Các model "phụ trợ" (router, multi-query, grading, guard) được khởi tạo
#    riêng bên trong từng module tương ứng — không cần load ở đây.
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Khởi tạo AI...")
def load_pipeline_components(_vectordb, _all_chunks):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )
    hybrid_retriever = HybridRetriever(_vectordb, _all_chunks)
    reranker = CrossEncoderReranker()
    return llm, hybrid_retriever, reranker


# ──────────────────────────────────────────────
# 6. Streamlit UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbox", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbox")
st.caption("Hỏi đáp thông tin từ tài liệu nội bộ — Pipeline 5 giai đoạn")

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

vectordb, all_chunks = load_vectordb_and_chunks()
llm, hybrid_retriever, reranker = load_pipeline_components(vectordb, all_chunks)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Cài đặt")
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()
    if st.button("🔄 Tải lại tài liệu"):
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("**LLM chính:** Llama 3.3 70B (Groq)")
    st.markdown("**LLM phụ trợ:** Llama 3.1 8B Instant (Groq)")
    st.markdown("**Embedding:** multilingual-e5-small")
    st.markdown("**Reranker:** BGE-reranker-v2-m3 (local)")
    st.markdown("**Guard:** Llama Guard 3 8B (Groq)")
    st.markdown("---")
    st.markdown(
        "**Pipeline:**\n"
        "1️⃣ Query Transformation\n"
        "2️⃣ Hybrid Retrieval (Vector+BM25+RRF)\n"
        "3️⃣ Reranking\n"
        "4️⃣ Corrective RAG\n"
        "5️⃣ Generation + Guard"
    )
    st.success("✅ Sẵn sàng")

    st.markdown("---")
    st.subheader("📊 Quản lý")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            log_bytes = f.read()
        num_rows = max(log_bytes.count(b"\n") - 1, 0)  # trừ dòng header
        st.caption(f"Đã ghi nhận {num_rows} lượt hỏi.")
        st.download_button(
            "⬇️ Tải log hỏi-đáp (CSV)",
            data=log_bytes,
            file_name=f"chat_log_{date.today()}.csv",
            mime="text/csv",
        )
    else:
        st.caption("Chưa có log nào được ghi.")

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
            result = answer_question(llm, hybrid_retriever, reranker, user_input, history)

        # Ghi log quản lý ngay khi có kết quả — không phụ thuộc phần render UI
        # bên dưới, nên dù UI lỗi (hiếm) thì log vẫn được lưu đầy đủ.
        log_turn(user_input, result.standalone_query, result)

        st.markdown(result.answer)

        if result.low_confidence:
            st.info(f"⚠️ Tài liệu nội bộ có thể chưa đủ thông tin: {result.low_confidence_reason}")

        if result.guard_flagged:
            st.warning("⚠️ Câu trả lời đã được kiểm duyệt bởi Llama Guard.")

        if result.sub_queries:
            with st.expander("🔍 Câu hỏi mở rộng (Multi-Query)"):
                for q in result.sub_queries:
                    st.markdown(f"- {q}")

        if result.sources:
            with st.expander("📄 Nguồn tham khảo"):
                seen = set()
                for doc in result.sources:
                    src     = doc.metadata.get("source", "unknown")
                    src_label = os.path.basename(src)
                    section = doc.metadata.get("section", "")
                    chapter = doc.metadata.get("chapter", "")
                    score   = doc.metadata.get("rerank_score", None)
                    key     = f"{src}::{section}"
                    if key not in seen:
                        label = f"{chapter} > {section}" if chapter and chapter != section else section
                        score_label = f" _(điểm rerank: {score})_" if score is not None else ""
                        st.markdown(f"- `{src_label}` — {label}{score_label}")
                        seen.add(key)

    st.session_state.messages.append({"role": "assistant", "content": result.answer})
