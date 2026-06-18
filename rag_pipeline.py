r"""
rag_pipeline.py
────────────────────────────────────────────────────────────────
Logic RAG 5 giai đoạn, tách riêng khỏi UI Streamlit (chatbox.py).

    GIAI ĐOẠN 1  Query Transformation   → query_transform.py (Router + Multi-Query)
    GIAI ĐOẠN 2  Hybrid Retrieval       → hybrid_retriever.py (Vector + BM25 + RRF)
    GIAI ĐOẠN 3  Reranking              → reranker.py (BGE Cross-Encoder, local CPU)
    GIAI ĐOẠN 4  Corrective RAG (CRAG)  → corrective_rag.py (chỉ chấm điểm, KHÔNG web search)
    GIAI ĐOẠN 5  Generation + Guard     → generation.py (Sinh câu trả lời + Llama Guard)

File này là "nhạc trưởng" (orchestrator): ghép 5 module trên thành 1 hàm
`answer_question()` duy nhất để chatbox.py gọi.

GHI CHÚ: bước Web Search bổ sung ở Giai đoạn 4 đã được BỎ theo yêu cầu
(gây chậm/thừa cho nhu cầu hiện tại). Khi tài liệu nội bộ bị chấm THIẾU/SAI,
pipeline không tự đi tìm thêm trên mạng nữa — chỉ đánh cờ `low_confidence`
để UI cảnh báo người dùng rằng câu trả lời dựa trên tài liệu có thể chưa đủ.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from query_transform import transform_query
from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker
from corrective_rag import grade_documents
from generation import generate_answer, check_guardrail


# ──────────────────────────────────────────────────────────────
# Kết quả trả về cho UI — gói đầy đủ thông tin từng giai đoạn để
# hiển thị debug/trace ngoài Streamlit nếu cần (giai đoạn nào chạy,
# tài liệu có đủ tin cậy không, có bị Guard chặn không...).
# ──────────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    answer: str
    sources: list[Document] = field(default_factory=list)
    low_confidence: bool = False
    low_confidence_reason: str = ""
    guard_flagged: bool = False
    sub_queries: list[str] = field(default_factory=list)
    standalone_query: str = ""    # câu hỏi sau Giai đoạn 1 (Router) — dùng để ghi log,
                                   # tránh phải gọi lại transform_query() ở chatbox.py


def answer_question(
    llm: ChatGroq,
    hybrid_retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    user_input: str,
    history: list[dict],
) -> PipelineResult:
    """
    Chạy toàn bộ 5 giai đoạn cho 1 câu hỏi người dùng.
    Đây là hàm duy nhất chatbox.py cần gọi.
    """

    # ── GIAI ĐOẠN 1: Query Transformation ──
    # Router viết lại câu hỏi thành stand-alone (gộp lịch sử hội thoại),
    # rồi Multi-Query sinh thêm 3 câu hỏi đồng nghĩa để bao vùng tìm kiếm.
    standalone_query, sub_queries = transform_query(llm, user_input, history)

    # ── GIAI ĐOẠN 2: Hybrid Retrieval ──
    # Vector Search (ngữ nghĩa) + Lexical Search BM25 (từ khóa chính xác),
    # đồng bộ kết quả từ N câu hỏi (gốc + sub_queries) bằng Reciprocal Rank Fusion.
    all_queries = [standalone_query] + sub_queries
    fused_docs = hybrid_retriever.retrieve(all_queries)

    # ── GIAI ĐOẠN 3: Reranking ──
    # Cross-Encoder đánh giá độ khớp sâu giữa câu hỏi – từng chunk,
    # cắt tỉa: chỉ giữ chunk có điểm rerank > ngưỡng, lọc bỏ hoàn toàn phần còn lại.
    reranked_docs = reranker.rerank(standalone_query, fused_docs)

    # ── GIAI ĐOẠN 4: Corrective RAG (CRAG, bản rút gọn) ──
    # Chấm điểm: tài liệu có đủ/đúng để trả lời không?
    #   ĐỦ        → đi tiếp dùng LLM sinh câu trả lời như bình thường.
    #   THIẾU/SAI → KHÔNG tra web nữa, chỉ đánh cờ cảnh báo cho UI hiển thị,
    #               LLM ở Giai đoạn 5 vẫn cố trả lời nhưng sẽ được nhắc rõ
    #               để trung thực nói "không tìm thấy" nếu thực sự thiếu.
    grade = grade_documents(llm, standalone_query, reranked_docs)
    low_confidence = grade.verdict != "DU"

    # ── GIAI ĐOẠN 5: Generation + Guardrails ──
    # Sinh câu trả lời kèm trích dẫn nguồn, rồi Llama Guard kiểm tra
    # câu trả lời có an toàn / không bịa đặt (hallucination) trước khi trả về.
    raw_answer = generate_answer(llm, standalone_query, reranked_docs, history)
    safe_answer, guard_flagged = check_guardrail(raw_answer, reranked_docs)

    return PipelineResult(
        answer=safe_answer,
        sources=reranked_docs,
        low_confidence=low_confidence,
        low_confidence_reason=grade.reason if low_confidence else "",
        guard_flagged=guard_flagged,
        sub_queries=sub_queries,
        standalone_query=standalone_query,
    )