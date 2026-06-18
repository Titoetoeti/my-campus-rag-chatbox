r"""
query_transform.py
────────────────────────────────────────────────────────────────
GIAI ĐOẠN 1: TỐI ƯU CÂU HỎI (Query Transformation)

  1. Chat History Router : tổng hợp lịch sử hội thoại + câu hỏi hiện tại
                            → viết lại thành 1 câu hỏi độc lập (stand-alone).
                            Cần thiết vì câu hỏi người dùng thường có đại từ
                            tham chiếu ("nó", "cái đó", "vậy còn...") mà nếu
                            đem thẳng đi tìm kiếm sẽ không ra kết quả đúng.

  2. Query Expansion (Multi-Query) : từ câu hỏi độc lập, sinh ra 3 câu hỏi
                            đồng nghĩa/diễn đạt khác để bao vùng tìm kiếm
                            rộng hơn (người dùng có thể hỏi 1 ý nhưng dùng
                            từ ngữ khác với tài liệu gốc).
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Mô hình nhỏ, nhanh, rẻ — dùng riêng cho các bước "phụ trợ" (router, multi-query,
# grading, guard ở các module khác) để không tốn quota của model chính (trả lời).
UTILITY_MODEL = "llama-3.1-8b-instant"

NUM_SUB_QUERIES = 3


def _build_history_text(history: list[dict], max_turns: int = 6) -> str:
    """Lấy tối đa `max_turns` lượt hội thoại gần nhất, format dạng text đơn giản."""
    if not history:
        return "(Không có lịch sử — đây là câu hỏi đầu tiên)"

    recent = history[-max_turns:]
    lines = []
    for m in recent:
        role = "Người dùng" if m["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    "Bạn là bộ định tuyến câu hỏi (Chat History Router). Nhiệm vụ: dựa vào "
    "LỊCH SỬ HỘI THOẠI bên dưới, viết lại CÂU HỎI HIỆN TẠI thành một câu hỏi "
    "độc lập (stand-alone), đầy đủ ngữ cảnh, không cần biết lịch sử mới hiểu được.\n\n"
    "QUY TẮC:\n"
    "- Nếu câu hỏi hiện tại đã rõ nghĩa, độc lập → giữ nguyên, không cần sửa.\n"
    "- Nếu có đại từ tham chiếu ('nó', 'cái đó', 'vậy còn', 'thế thì'...) → thay "
    "bằng danh từ cụ thể lấy từ lịch sử.\n"
    "- KHÔNG trả lời câu hỏi, CHỈ viết lại câu hỏi.\n"
    "- Trả về DUY NHẤT câu hỏi đã viết lại, không thêm giải thích, không thêm tiền tố.\n\n"
    "LỊCH SỬ HỘI THOẠI:\n{history}\n\n"
    "CÂU HỎI HIỆN TẠI: {question}\n\n"
    "CÂU HỎI ĐỘC LẬP:"
)

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "Bạn là bộ mở rộng câu hỏi (Query Expansion). Từ CÂU HỎI GỐC bên dưới, hãy "
    "sinh ra {n} câu hỏi khác có CÙNG Ý NGHĨA nhưng diễn đạt khác đi (đồng nghĩa, "
    "đổi cấu trúc câu, dùng từ khác) để tăng khả năng tìm đúng tài liệu liên quan.\n\n"
    "QUY TẮC:\n"
    "- Mỗi câu hỏi trên 1 dòng, không đánh số, không markdown, không giải thích.\n"
    "- Giữ nguyên ngôn ngữ (tiếng Việt) và giữ đúng chủ đề câu hỏi gốc.\n"
    "- Không lặp lại y nguyên câu hỏi gốc.\n\n"
    "CÂU HỎI GỐC: {question}\n\n"
    "{n} CÂU HỎI ĐỒNG NGHĨA:"
)


def _get_utility_llm(api_key: str) -> ChatGroq:
    return ChatGroq(model=UTILITY_MODEL, api_key=api_key, temperature=0)


def rewrite_standalone(llm_main: ChatGroq, question: str, history: list[dict]) -> str:
    """Giai đoạn 1.1 — Chat History Router."""
    if not history:
        # Không có lịch sử → không cần viết lại, tránh tốn 1 lượt gọi LLM vô ích.
        return question.strip()

    utility_llm = _get_utility_llm(llm_main.groq_api_key.get_secret_value())
    chain = _ROUTER_PROMPT | utility_llm | StrOutputParser()
    rewritten = chain.invoke({
        "history": _build_history_text(history),
        "question": question,
    }).strip()

    # An toàn: nếu LLM trả về rỗng hoặc lỗi format → fallback dùng câu hỏi gốc.
    return rewritten if rewritten else question.strip()


def expand_queries(llm_main: ChatGroq, standalone_question: str, n: int = NUM_SUB_QUERIES) -> list[str]:
    """Giai đoạn 1.2 — Multi-Query Expansion."""
    utility_llm = _get_utility_llm(llm_main.groq_api_key.get_secret_value())
    chain = _MULTI_QUERY_PROMPT | utility_llm | StrOutputParser()
    raw = chain.invoke({"question": standalone_question, "n": n})

    lines = [ln.strip(" -•0123456789.\t") for ln in raw.split("\n")]
    queries = [ln for ln in lines if ln and ln.lower() != standalone_question.lower()]
    return queries[:n]


def transform_query(llm_main: ChatGroq, question: str, history: list[dict]) -> tuple[str, list[str]]:
    """
    Hàm tổng hợp Giai đoạn 1: trả về (câu hỏi độc lập, danh sách câu hỏi mở rộng).
    Mọi lỗi phụ trợ (router/expansion) đều fallback an toàn về câu hỏi gốc,
    không bao giờ làm gãy pipeline chỉ vì bước "tối ưu" này thất bại.
    """
    try:
        standalone = rewrite_standalone(llm_main, question, history)
    except Exception:
        standalone = question.strip()

    try:
        sub_queries = expand_queries(llm_main, standalone)
    except Exception:
        sub_queries = []

    return standalone, sub_queries