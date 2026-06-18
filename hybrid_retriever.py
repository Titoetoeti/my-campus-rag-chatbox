r"""
hybrid_retriever.py
────────────────────────────────────────────────────────────────
GIAI ĐOẠN 2: TÌM KIẾM HỖN HỢP (Hybrid Retrieval)

  1. Vector Search  : tìm theo ngữ nghĩa (embedding multilingual-e5-small,
                       đã có sẵn từ trước — không đổi).
  2. Lexical Search  : tìm theo từ khóa chính xác bằng BM25 (rank_bm25).
                       Bắt được các trường hợp Vector Search bỏ lỡ, ví dụ
                       người dùng hỏi đúng số "Điều 15" hay 1 thuật ngữ
                       pháp lý cụ thể mà embedding không phân biệt rõ.
  3. Reciprocal Rank Fusion (RRF) : đồng bộ + xếp hạng lại kết quả từ
                       2 nguồn trên (chạy cho N câu hỏi từ Multi-Query),
                       gộp thành 1 danh sách duy nhất theo công thức:

                           RRF_score(d) = Σ  1 / (k + rank_i(d))

                       với rank_i(d) là vị trí của tài liệu d trong danh
                       sách kết quả thứ i (k là hằng số làm mượt, mặc định 60).
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import defaultdict

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

RRF_K = 60                 # hằng số làm mượt RRF (giá trị chuẨn dùng phổ biến: 60)
VECTOR_TOP_K = 8           # số chunk lấy về mỗi lượt Vector Search
BM25_TOP_K = 8             # số chunk lấy về mỗi lượt Lexical Search
FUSED_TOP_K = 15           # số chunk giữ lại sau khi fusion, trước khi đưa qua Reranker


def _doc_key(doc: Document) -> str:
    """
    Khóa định danh duy nhất cho 1 chunk, dùng để gộp kết quả trùng nhau
    giữa Vector Search và BM25 (vì chúng trả về object Document khác nhau
    cho cùng 1 nội dung).
    """
    src = doc.metadata.get("source", "")
    section = doc.metadata.get("section", "")
    # Thêm 40 ký tự đầu nội dung để phân biệt các chunk bị split nhỏ
    # cùng section (trường hợp _split_long cắt 1 Điều dài thành nhiều phần).
    return f"{src}::{section}::{doc.page_content[:40]}"


class HybridRetriever:
    """
    Bọc 2 retriever (Vector FAISS có sẵn + BM25 mới khởi tạo) thành 1 giao diện
    duy nhất: nhận N câu hỏi → trả về danh sách Document đã fusion + xếp hạng.
    """

    def __init__(self, vectordb: FAISS, all_documents: list[Document]):
        self.vectordb = vectordb
        # BM25Retriever cần load lại toàn bộ corpus vào RAM dạng tokenized,
        # khác với FAISS (đã index sẵn) — nên khởi tạo 1 lần, tái dùng nhiều câu hỏi.
        self.bm25 = BM25Retriever.from_documents(all_documents)
        self.bm25.k = BM25_TOP_K

    def _vector_search(self, query: str) -> list[Document]:
        return self.vectordb.similarity_search(query, k=VECTOR_TOP_K)

    def _lexical_search(self, query: str) -> list[Document]:
        return self.bm25.invoke(query)

    def _rrf_fuse(self, ranked_lists: list[list[Document]]) -> list[Document]:
        """
        Reciprocal Rank Fusion trên nhiều danh sách kết quả (mỗi danh sách
        đến từ 1 retriever, cho 1 câu hỏi). Trả về 1 danh sách duy nhất,
        sắp xếp giảm dần theo điểm RRF tổng hợp.
        """
        scores: dict[str, float] = defaultdict(float)
        doc_lookup: dict[str, Document] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                key = _doc_key(doc)
                scores[key] += 1.0 / (RRF_K + rank + 1)
                doc_lookup.setdefault(key, doc)

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_lookup[k] for k in sorted_keys[:FUSED_TOP_K]]

    def retrieve(self, queries: list[str]) -> list[Document]:
        """
        Nhận danh sách câu hỏi (câu hỏi gốc + các câu hỏi mở rộng từ Giai đoạn 1),
        chạy Vector + Lexical Search cho TỪNG câu hỏi, rồi RRF gộp tất cả lại.
        """
        ranked_lists = []
        for q in queries:
            ranked_lists.append(self._vector_search(q))
            ranked_lists.append(self._lexical_search(q))

        return self._rrf_fuse(ranked_lists)