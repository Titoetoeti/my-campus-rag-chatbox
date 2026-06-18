r"""
corrective_rag.py
────────────────────────────────────────────────────────────────
GIAI ĐOẠN 4: THẨM ĐỊNH (Corrective RAG - CRAG, bản rút gọn)

  Kiểm tra: "Các tài liệu này có thực sự đủ để trả lời câu hỏi?"

    ĐỦ        → đi tiếp dùng LLM sinh câu trả lời (Giai đoạn 5).
    THIẾU/SAI → KHÔNG còn tự động tra web bổ sung (đã bỏ theo yêu cầu, vì
                bước này làm chậm/thừa cho nhu cầu hiện tại). Thay vào đó,
                chỉ đánh cờ cảnh báo để Giai đoạn 5 biết và trả lời trung
                thực "không tìm thấy trong tài liệu" thay vì bịa.

  Lưu ý: nếu sau này muốn bật lại web search, chỉ cần viết lại 1 hàm
  `web_search_fallback()` ở đây và gọi nó trong rag_pipeline.py — phần
  grading bên dưới không cần đổi gì.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

UTILITY_MODEL = "llama-3.1-8b-instant"


@dataclass
class GradeResult:
    verdict: str   # "DU" (đủ) | "THIEU" (thiếu) | "SAI" (sai/lệch chủ đề)
    reason: str    # lý do ngắn, dùng hiển thị debug cho người dùng


_GRADE_PROMPT = ChatPromptTemplate.from_template(
    "Bạn là bộ thẩm định chất lượng tài liệu (Corrective RAG Grader).\n"
    "Nhiệm vụ: đọc CÂU HỎI và các ĐOẠN TÀI LIỆU được tìm thấy, đánh giá xem "
    "tài liệu có ĐỦ THÔNG TIN để trả lời chính xác câu hỏi không.\n\n"
    "Trả về DUY NHẤT 1 trong 3 từ sau (không thêm gì khác):\n"
    "- DU     : tài liệu đủ thông tin liên quan để trả lời.\n"
    "- THIEU  : tài liệu có liên quan nhưng thiếu chi tiết quan trọng.\n"
    "- SAI    : tài liệu không liên quan / lệch chủ đề câu hỏi.\n\n"
    "CÂU HỎI: {question}\n\n"
    "ĐOẠN TÀI LIỆU:\n{context}\n\n"
    "KẾT QUẢ (chỉ 1 từ DU/THIEU/SAI):"
)


def _format_context(documents: list[Document], max_chars: int = 3000) -> str:
    """Ghép nội dung các chunk lại, cắt giới hạn ký tự để không tốn token chấm điểm."""
    text = "\n---\n".join(d.page_content for d in documents)
    return text[:max_chars]


def grade_documents(llm: ChatGroq, question: str, documents: list[Document]) -> GradeResult:
    """
    Chấm điểm danh sách tài liệu đã rerank ở Giai đoạn 3.
    Nếu danh sách rỗng (không tìm được gì) → chắc chắn THIEU, không cần gọi LLM.
    """
    if not documents:
        return GradeResult(verdict="THIEU", reason="Không tìm thấy tài liệu nội bộ liên quan.")

    try:
        utility_llm = ChatGroq(
            model=UTILITY_MODEL,
            api_key=llm.groq_api_key.get_secret_value(),
            temperature=0,
        )
        chain = _GRADE_PROMPT | utility_llm | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "context": _format_context(documents),
        }).strip().upper()

        if "DU" in raw and "THIEU" not in raw:
            return GradeResult(verdict="DU", reason="Tài liệu nội bộ đủ thông tin.")
        elif "SAI" in raw:
            return GradeResult(verdict="SAI", reason="Tài liệu nội bộ không liên quan đến câu hỏi.")
        else:
            return GradeResult(verdict="THIEU", reason="Tài liệu nội bộ thiếu chi tiết quan trọng.")

    except Exception as e:
        # Lỗi chấm điểm (timeout, lỗi API...) → coi như DU để không chặn câu trả lời
        # chỉ vì bước phụ trợ này lỗi (fail-open, tránh app bị treo/báo lỗi oan).
        return GradeResult(verdict="DU", reason=f"Bỏ qua chấm điểm do lỗi ({e}).")