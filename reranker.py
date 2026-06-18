r"""
reranker.py
────────────────────────────────────────────────────────────────
GIAI ĐOẠN 3: SÀNG LỌC & TÁI XẾP HẠNG (Post-Processing)

  1. Cross-Encoder Reranker : dùng BAAI/bge-reranker-v2-m3 (chạy CPU, local,
                       không cần API key) để đánh giá ĐỘ KHỚP THỰC SỰ giữa
                       câu hỏi và từng chunk. Khác với Vector/BM25 Search
                       (chỉ so sánh embedding/từ khóa độc lập), Cross-Encoder
                       đọc CẢ CÂU HỎI + CẢ CHUNK cùng lúc → chính xác hơn
                       nhiều nhưng chậm hơn, nên chỉ áp dụng sau khi đã
                       lọc xuống danh sách ngắn (FUSED_TOP_K) ở Giai đoạn 2.

                       VỀ VIỆC CHỌN MODEL: đã thử đổi sang bge-reranker-base
                       (nhẹ hơn) để giảm thời gian khởi tạo, nhưng base CHỈ
                       được BAAI khuyến nghị cho tiếng Trung/Anh — không phải
                       model đa ngôn ngữ. v2-m3 mới là bản có nền BGE-M3 hỗ
                       trợ 100+ ngôn ngữ (gồm tiếng Việt) nên giữ lại v2-m3
                       để không đánh đổi chất lượng rerank tiếng Việt. Thời
                       gian khởi tạo lâu (model ~2.27GB) được xử lý bằng cách
                       tải song song nhanh hơn (hf_transfer) + log tiến trình
                       rõ ràng bên dưới, KHÔNG bằng cách đổi sang model nhẹ
                       nhưng kém phù hợp ngôn ngữ.

  2. Cắt tỉa (Trimming) : chỉ giữ lại các chunk có điểm Rerank > ngưỡng
                       (RERANK_THRESHOLD). Các chunk dưới ngưỡng bị loại bỏ
                       hoàn toàn — không đưa vào prompt sinh câu trả lời,
                       giúp giảm nhiễu (irrelevant context) và tiết kiệm token.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os

from langchain_core.documents import Document

# Bật hf_transfer nếu đã cài (pip install hf_transfer) — tăng tốc tải model
# từ HuggingFace đáng kể trên đường truyền tốt. Không bắt buộc: nếu chưa cài,
# dòng này chỉ là 1 biến môi trường vô hại, sentence-transformers tự fallback
# về downloader mặc định, không lỗi gì cả.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_THRESHOLD = 0.30     # điểm rerank tối thiểu để giữ lại 1 chunk (thang 0-1 sau sigmoid)
RERANK_TOP_K = 6            # số chunk tối đa giữ lại sau rerank (đưa vào prompt LLM)
MIN_KEEP = 2                # luôn giữ tối thiểu N chunk điểm cao nhất, dù không qua ngưỡng,
                             # để tránh CRAG (giai đoạn 4) lúc nào cũng phải báo "thiếu thông tin".


class CrossEncoderReranker:
    """
    Bọc model Cross-Encoder (sentence-transformers) thành 1 hàm rerank() duy nhất.
    Model được load 1 lần (nặng ~2.27GB, lần đầu tải có thể mất vài phút tùy
    mạng — XEM GHI CHÚ TỐC ĐỘ ở cuối file) rồi tái sử dụng cho mọi câu hỏi,
    không load lại mỗi lần gọi.
    """

    def __init__(self):
        # Import trong __init__ (không import ở top-level) để chatbox.py có thể
        # khởi tạo các phần khác trước, lỗi thiếu thư viện sentence-transformers
        # (nếu có) sẽ chỉ raise đúng lúc thực sự cần model này.
        import time
        from sentence_transformers import CrossEncoder

        print(f"[reranker] Bắt đầu load {RERANK_MODEL} (lần đầu sẽ tải ~2.27GB)...")
        t0 = time.time()
        self.model = CrossEncoder(RERANK_MODEL, max_length=512, device="cpu")
        elapsed = time.time() - t0
        print(f"[reranker] Đã load {RERANK_MODEL} trong {elapsed:.1f}s")

        if elapsed > 300:
            print(
                "[reranker] CẢNH BÁO: load mất hơn 5 phút — rất có thể do mạng tới "
                "HuggingFace chậm, không phải do code. Xem ghi chú cuối file reranker.py "
                "để biết cách tải trước (pre-download) hoặc dùng cache thư mục."
            )

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Chấm điểm từng document với query, sắp xếp giảm dần theo điểm,
        cắt tỉa theo ngưỡng + top-k, trả về danh sách Document đã lọc
        (mỗi Document được gắn thêm metadata "rerank_score" để debug/hiển thị).
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        raw_scores = self.model.predict(pairs)

        # CrossEncoder trả logit thô → chuẩn hóa sigmoid về [0, 1] để áp ngưỡng dễ hiểu.
        scores = [1 / (1 + pow(2.718281828, -s)) for s in raw_scores]

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        kept = []
        for doc, score in scored:
            doc.metadata["rerank_score"] = round(float(score), 4)
            if score > RERANK_THRESHOLD or len(kept) < MIN_KEEP:
                kept.append(doc)
            if len(kept) >= RERANK_TOP_K:
                break

        return kept


# ──────────────────────────────────────────────────────────────
# GHI CHÚ TỐC ĐỘ — đọc nếu lần khởi tạo đầu tiên quá lâu (>10 phút)
# ──────────────────────────────────────────────────────────────
#
# Model ~2.27GB tải qua mạng tốt (~10-20 Mbps) thường mất 3-8 phút, KHÔNG phải
# 25-30 phút. Nếu máy bạn bị lâu bất thường, nhiều khả năng nhất là:
#
# 1. Mạng tới huggingface.co bị chậm/chập (phổ biến nhất ở VN do định tuyến
#    quốc tế). Cách kiểm tra nhanh: mở terminal khác, gõ
#       ping huggingface.co
#    nếu độ trễ cao hoặc rớt gói liên tục → đúng nguyên nhân là mạng.
#
# 2. Để TĂNG TỐC tải, cài thêm gói hf_transfer (dùng giao thức tải nhiều luồng):
#       pip install hf_transfer
#    Code ở trên đã tự động bật cờ HF_HUB_ENABLE_HF_TRANSFER=1 nếu gói này có sẵn.
#
# 3. Để TRÁNH phải tải lại mỗi khi đổi máy/môi trường, model sau khi tải sẽ
#    được cache tại:  C:\Users\<bạn>\.cache\huggingface\hub
#    Nếu cần triển khai ở máy khác, có thể copy thẳng thư mục cache này sang
#    để khởi tạo gần như tức thì (không cần tải lại từ mạng).
# ──────────────────────────────────────────────────────────────