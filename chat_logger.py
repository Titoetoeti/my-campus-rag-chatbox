r"""
chat_logger.py
────────────────────────────────────────────────────────────────
GHI LOG QUẢN LÝ — lưu lại toàn bộ lượt hỏi-đáp ra file CSV cục bộ.

Lý do cần file này: Groq Console chỉ hiển thị metadata (request ID, số
token, latency) cho mục đích giám sát hệ thống, KHÔNG lưu lại nội dung
câu hỏi/câu trả lời vì lý do bảo mật/quyền riêng tư phía họ. Muốn xem lại
"ai hỏi gì, trả lời ra sao, dựa vào nguồn nào" thì phải tự ghi log ở phía
ứng dụng của mình — đó là việc file này làm.

THIẾT KẾ: mỗi lượt hỏi = 1 dòng CSV, mở được trực tiếp bằng Excel/Notepad.
Các cột chính (thời gian, câu hỏi, trả lời, độ tin cậy, nguồn cao điểm nhất)
hiện ngay khi mở file để xem nhanh. Cột `chi_tiet_json` chứa đầy đủ thông
tin từng nguồn + điểm rerank của từng nguồn (dạng JSON) — không bắt buộc
phải đọc nếu chỉ cần xem nhanh, nhưng vẫn có sẵn khi cần đào sâu hoặc xử lý
lại bằng code (ví dụ tính trung bình điểm rerank theo tuần).

FILE LOG: logs/chat_log.csv (tự tạo thư mục + file nếu chưa có, tự thêm
header 1 lần duy nhất). Mỗi lượt hỏi append 1 dòng — không bao giờ đọc lại
toàn bộ file để ghi, nên không bị chậm dần khi file lớn lên.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "chat_log.csv")

# Thứ tự cột cố định — KHÔNG đổi thứ tự sau khi đã có dữ liệu cũ trong file,
# vì Excel/pandas đọc CSV theo đúng vị trí cột, đổi thứ tự sẽ làm lệch dữ
# liệu cũ. Muốn thêm cột mới, luôn thêm vào CUỐI danh sách.
CSV_COLUMNS = [
    "thoi_gian",            # ISO format, dễ sort theo thời gian trong Excel
    "cau_hoi",              # câu hỏi gốc người dùng gõ
    "cau_hoi_doc_lap",      # câu hỏi sau Giai đoạn 1 (Router viết lại, nếu có lịch sử)
    "cau_tra_loi",          # câu trả lời cuối cùng (sau Guard)
    "do_tin_cay",           # "DU" | "THIEU" | "SAI" — kết quả chấm điểm CRAG
    "bi_chan_boi_guard",    # True/False — Llama Guard có chặn câu trả lời gốc không
    "so_nguon_trich_dan",   # số lượng chunk tài liệu được dùng làm context
    "nguon_diem_cao_nhat",  # tên file + điểm rerank của nguồn tốt nhất (xem nhanh)
    "chi_tiet_json",        # JSON đầy đủ: toàn bộ nguồn + điểm rerank từng nguồn
]


def _ensure_log_file():
    """Tạo thư mục logs/ và file CSV kèm header nếu chưa tồn tại."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8-sig") as f:
            # utf-8-sig (có BOM) để Excel trên Windows tự nhận đúng encoding
            # tiếng Việt có dấu, không bị hiện chữ lỗi khi mở trực tiếp.
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def _best_source_label(sources: list) -> str:
    """Tìm nguồn có điểm rerank cao nhất, trả về dạng text ngắn để xem nhanh."""
    if not sources:
        return "(không có nguồn)"

    best = max(sources, key=lambda d: d.metadata.get("rerank_score", 0))
    name = os.path.basename(best.metadata.get("source", "?"))
    section = best.metadata.get("section", "")
    score = best.metadata.get("rerank_score", "?")
    return f"{name} | {section} (điểm: {score})"


def _sources_to_detail(sources: list) -> str:
    """Chuyển toàn bộ danh sách nguồn thành JSON gọn để lưu trong 1 cột."""
    detail = []
    for d in sources:
        detail.append({
            "file": os.path.basename(d.metadata.get("source", "?")),
            "chuong": d.metadata.get("chapter", ""),
            "dieu": d.metadata.get("section", ""),
            "diem_rerank": d.metadata.get("rerank_score"),
            "noi_dung_rut_gon": d.page_content[:200],
        })
    return json.dumps(detail, ensure_ascii=False)


def log_turn(
    user_question: str,
    standalone_question: str,
    result,
) -> None:
    """
    Ghi 1 dòng log cho 1 lượt hỏi-đáp. Gọi hàm này ngay sau khi có `result`
    từ rag_pipeline.answer_question(), trước hoặc sau khi hiển thị UI đều
    được — không phụ thuộc thứ tự.

    Lỗi ghi log (đĩa đầy, không có quyền ghi...) sẽ KHÔNG làm sập app —
    chỉ in cảnh báo ra console, vì đây là tác vụ phụ trợ (audit), không
    phải tác vụ chính (trả lời người dùng).
    """
    try:
        _ensure_log_file()

        row = [
            datetime.now().isoformat(timespec="seconds"),
            user_question,
            standalone_question,
            result.answer,
            "THIEU/SAI" if result.low_confidence else "DU",
            result.guard_flagged,
            len(result.sources),
            _best_source_label(result.sources),
            _sources_to_detail(result.sources),
        ]

        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    except Exception as e:
        # Không raise — ghi log lỗi không nên ảnh hưởng trải nghiệm người dùng.
        print(f"[chat_logger] Không ghi được log: {e}")