r"""
RAG Chatbox - Streamlit App (PDF-native version)
D:\simpleRAGchatbox\
    chatbox.py
    .env                <- GROQ_API_KEY=gsk_...
    paper\*.pdf         <- DOC TRUC TIEP PDF (khong can .txt nua)
    index\              <- tu tao khi chay lan dau
"""

import os
import re
import streamlit as st
from dotenv import load_dotenv

import pdfplumber
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
# 3. Đọc và chunk PDF theo cấu trúc văn bản pháp lý
#
# Chiến lược chunking:
#   - Tách theo Điều (Điều 1., Điều 2., ...) — đơn vị tri thức nhỏ nhất
#   - Mỗi chunk giữ nguyên tiêu đề Chương cha để tăng ngữ cảnh
#   - Chunk phụ lục (bảng vi phạm) tách riêng theo số thứ tự hàng
#   - Khi chunk quá dài (>1200 ký tự) → tách theo khoảng trắng đoạn
# ──────────────────────────────────────────────

# Pattern nhận diện cấu trúc văn bản pháp lý tiếng Việt
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

    # Tách thô theo Chương và Phụ lục
    split_points = []
    for m in RE_CHUONG.finditer(full_text):
        split_points.append(('chapter', m.start(), m.group(1)))
    for m in RE_PHAN.finditer(full_text):
        split_points.append(('annex', m.start(), m.group(1)))
    split_points.sort(key=lambda x: x[1])

    # Lấy phần giới thiệu trước Chương I
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

    # Xử lý từng Chương / Phụ lục
    for idx, (kind, start, title) in enumerate(split_points):
        end = split_points[idx + 1][1] if idx + 1 < len(split_points) else len(full_text)
        block = full_text[start:end].strip()

        if kind == 'chapter':
            current_chapter = title.strip()
            _chunk_chapter(block, current_chapter, source, docs)

        elif kind == 'annex':
            # Phụ lục bảng → chunk theo đoạn ngắn
            for chunk in _split_long(block):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": source, "chapter": title.strip(), "section": title.strip(), "chunk_type": "annex"}
                ))

    return docs


def _chunk_chapter(chapter_text: str, chapter_name: str, source: str, docs: list):
    """Chia một Chương thành các chunk theo Điều."""
    # Tìm vị trí các Điều trong Chương
    dieu_matches = list(RE_DIEU.finditer(chapter_text))

    if not dieu_matches:
        # Chương không có Điều riêng biệt → 1 chunk
        for chunk in _split_long(chapter_text):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": source, "chapter": chapter_name, "section": chapter_name, "chunk_type": "content"}
            ))
        return

    # Text trước Điều đầu tiên (tiêu đề chương)
    preamble = chapter_text[:dieu_matches[0].start()].strip()
    if preamble:
        docs.append(Document(
            page_content=f"{chapter_name}\n{preamble}",
            metadata={"source": source, "chapter": chapter_name, "section": chapter_name + " - Mở đầu", "chunk_type": "content"}
        ))

    # Từng Điều
    for i, m in enumerate(dieu_matches):
        dieu_title = m.group(1).strip()
        dieu_start = m.start()
        dieu_end   = dieu_matches[i + 1].start() if i + 1 < len(dieu_matches) else len(chapter_text)
        dieu_body  = chapter_text[dieu_start:dieu_end].strip()

        # Tiêu đề Điều + tên Chương để tăng ngữ cảnh
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


# ──────────────────────────────────────────────
# 4. Vector DB — hỗ trợ cả PDF lẫn TXT (ưu tiên PDF)
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

    vectordb = FAISS.from_documents(all_chunks, embeddings)
    vectordb.save_local(INDEX_DIR)
    return vectordb


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
# 5. LLM + Retriever
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
        search_kwargs={"k": 6}
    )
    return llm, retriever


def dedupe_by_section(docs: list[Document]) -> list[Document]:
    """Loại bỏ chunk trùng section, ưu tiên giữ chunk content."""
    seen = {}
    for doc in docs:
        key   = doc.metadata.get("section", "")
        ctype = doc.metadata.get("chunk_type", "content")
        if key not in seen:
            seen[key] = doc
        elif ctype == "content":
            seen[key] = doc
    return list(seen.values())


# ──────────────────────────────────────────────
# 6. Hàm trả lời
# ──────────────────────────────────────────────
def get_answer(llm, retriever, user_input, history):
    raw_docs = retriever.invoke(user_input)
    docs     = dedupe_by_section(raw_docs)

    context_parts = []
    for i, d in enumerate(docs):
        src     = os.path.basename(d.metadata.get("source", "unknown"))
        chapter = d.metadata.get("chapter", "")
        section = d.metadata.get("section", "")
        label   = f"{src} | {chapter} | {section}" if chapter != section else f"{src} | {section}"
        context_parts.append(f"[Đoạn {i+1} | {label}]\n{d.page_content}")
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
        "4. Nếu câu hỏi liên quan đến quy định/điều cấm → trích dẫn rõ số Điều và nội dung.\n\n"
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
# 7. Streamlit UI
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
    if st.button("🔄 Tải lại tài liệu"):
        import shutil
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("**LLM:** Llama 3.3 70B (Groq)")
    st.markdown("**Embedding:** multilingual-e5-small")
    st.markdown("**Chunking:** Chương → Điều (PDF-native)")
    st.markdown("**Nguồn:** PDF *(fallback: TXT)*")
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
                    chapter = doc.metadata.get("chapter", "")
                    key     = f"{src}::{section}"
                    if key not in seen:
                        label = f"{chapter} > {section}" if chapter and chapter != section else section
                        st.markdown(f"- `{src}` — {label}")
                        seen.add(key)

    st.session_state.messages.append({"role": "assistant", "content": answer})
