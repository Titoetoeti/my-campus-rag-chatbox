r"""
generation.py
────────────────────────────────────────────────────────────────
GIAI ĐOẠN 5: TỔNG HỢP & KIỂM DUYỆT (Generation & Guard)

  1. LLM sinh câu trả lời kèm Trích dẫn (Citations) : mỗi đoạn tài liệu được
                       đánh số + ghi rõ nguồn (tên file PDF), prompt yêu cầu
                       LLM trích dẫn [Đoạn n] khi dùng thông tin.

  2. Guardrails (Llama Guard qua Groq) : sau khi có câu trả lời, gửi qua model
                       `llama-guard-3-8b` (có sẵn trên Groq, không cần thêm
                       framework/API key mới) để kiểm tra nội dung có an toàn
                       không (unsafe content) và cảnh báo nếu câu trả lời có
                       dấu hiệu KHÔNG bám sát tài liệu (hallucination check
                       thủ công bằng cách yêu cầu LLM tự đối chiếu).
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

GUARD_MODEL = "llama-guard-3-8b"

_GENERATION_PROMPT = ChatPromptTemplate.from_template(
    "Bạn là trợ lý AI của tổ chức, chuyên trả lời câu hỏi dựa trên tài liệu nội bộ.\n\n"
    "NGUYÊN TẮC:\n"
    "1. Chỉ dùng thông tin trong 'NGỮ CẢNH TÀI LIỆU' bên dưới, không bịa thêm.\n"
    "2. Nếu không tìm thấy thông tin → nói rõ: 'Tôi không tìm thấy thông tin này trong tài liệu.'\n"
    "3. Trả lời ngắn gọn, rõ ràng, đúng trọng tâm câu hỏi.\n"
    "4. Nếu câu hỏi liên quan đến quy định/điều cấm → trích dẫn rõ số Điều và nội dung.\n"
    "5. Khi dùng thông tin từ 1 đoạn cụ thể, ghi chú [Đoạn n] ngay sau câu đó.\n\n"
    "NGỮ CẢNH TÀI LIỆU:\n{context}\n\n"
    "LỊCH SỬ HỘI THOẠI:\n{history}\n\n"
    "CÂU HỎI: {question}\n\n"
    "TRẢ LỜI:"
)


def _format_context(documents: list[Document]) -> str:
    context_parts = []
    for i, d in enumerate(documents):
        source = d.metadata.get("source", "unknown")
        src_label = os.path.basename(source)

        chapter = d.metadata.get("chapter", "")
        section = d.metadata.get("section", "")
        label = f"{src_label} | {chapter} | {section}" if chapter and chapter != section else f"{src_label} | {section}"

        context_parts.append(f"[Đoạn {i + 1} | {label}]\n{d.page_content}")
    return "\n\n".join(context_parts)


def _format_history(history: list[dict]) -> str:
    lines = []
    for m in history:
        role = "Người dùng" if m["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def generate_answer(
    llm: ChatGroq,
    question: str,
    documents: list[Document],
    history: list[dict],
) -> str:
    """Giai đoạn 5.1 — sinh câu trả lời kèm trích dẫn."""
    chain = _GENERATION_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "context": _format_context(documents),
        "history": _format_history(history),
        "question": question,
    })


def check_guardrail(answer: str, documents: list[Document]) -> tuple[str, bool]:
    """
    Giai đoạn 5.2 — Llama Guard kiểm duyệt câu trả lời.

    Trả về (câu_trả_lời_cuối_cùng, đã_bị_gắn_cờ).
    Nếu Llama Guard chấm "unsafe" → KHÔNG trả về câu trả lời gốc, thay bằng
    thông báo từ chối an toàn. Nếu Guard lỗi (timeout, hết quota...) → coi như
    "safe" (fail-open) để không làm sập trải nghiệm người dùng vì 1 bước phụ.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return answer, False

    try:
        guard = ChatGroq(model=GUARD_MODEL, api_key=api_key, temperature=0)
        # Llama Guard nhận message dạng hội thoại, chấm "an toàn" cho lượt trả lời
        # cuối cùng (assistant) dựa trên các hạng mục an toàn chuẩn (MLCommons).
        result = guard.invoke([
            {"role": "user", "content": "Nội dung câu trả lời của trợ lý AI cần được kiểm duyệt."},
            {"role": "assistant", "content": answer},
        ])
        verdict = result.content.strip().lower()

        if verdict.startswith("unsafe"):
            safe_message = (
                "⚠️ Câu trả lời ban đầu không vượt qua kiểm duyệt an toàn nội dung, "
                "nên đã được chặn lại. Vui lòng đặt câu hỏi theo cách khác hoặc liên hệ "
                "quản trị viên nếu bạn cho rằng đây là nhầm lẫn."
            )
            return safe_message, True

        return answer, False

    except Exception:
        # Fail-open: lỗi guard không nên chặn câu trả lời hợp lệ.
        return answer, False