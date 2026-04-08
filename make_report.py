#!/usr/bin/env python3
"""
Generate report.pdf for Assignment 3 (embeddings, LangGraph vs LangChain, chunking).

Usage:
  pip install fpdf2
  python make_report.py [--out report.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "I", 9)
        self.cell(
            0,
            8,
            "Assignment 3 - Autonomous Multi-Doc Financial Analyst",
            align="C",
        )
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section(self, title: str) -> None:
        self.set_font("Helvetica", "B", 14)
        self.multi_cell(0, 8, title)
        self.ln(2)
        self.set_font("Helvetica", "", 11)

    def body(self, text: str) -> None:
        self.multi_cell(0, 6, text)
        self.ln(4)


def build_pdf(out_path: Path) -> None:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.section("1. Introduction")
    pdf.body(
        "This project implements a retrieval-augmented financial analyst over Apple and Tesla "
        "SEC-style filings (10-K / consolidated statements). The pipeline answers factual questions "
        "with inline citations, routes queries to the correct company corpus, rewrites vague queries, "
        "grades relevance, and refuses when the filing does not support an answer."
    )

    pdf.section("2. Comparing embedding models")
    pdf.body(
        "The vector index is built with HuggingFace embeddings via langchain-huggingface. The default "
        "model is sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: it is compact, runs "
        "locally, and handles mixed Chinese/English queries (common in the benchmark). "
        "To compare a second model without overwriting indices, set EMBEDDING_MODEL to another "
        "sentence-transformers checkpoint (e.g. all-MiniLM-L6-v2 for a smaller English-biased model, "
        "or a larger model such as BAAI/bge-small-en-v1.5) and use a distinct EMBEDDING_TAG. "
        "Rebuild with: CHUNK_SIZE unchanged, run build_rag.py, then run evaluator.py --ci for both tags. "
        "Expect trade-offs: multilingual MiniLM improves retrieval on mixed-language questions; "
        "smaller English models can be faster but may miss non-English paraphrases; larger models "
        "often improve semantic precision at higher CPU/GPU cost."
    )

    pdf.section("3. LangGraph vs LangChain ReAct")
    pdf.body(
        "Task A uses a classic ReAct-style agent (single tool loop over a retriever). "
        "The main system uses LangGraph: (1) an LLM router selects apple / tesla / both / none; "
        "(2) retrieval runs with MMR, query-specific boosts, and anchor chunks for fragile table rows; "
        "(3) a binary relevance grader decides whether to proceed; (4) on failure, a rewriter refines "
        "the query and retrieval repeats; (5) a final generator answers in English with honesty rules "
        "and source tags. Compared to ReAct, the graph separates concerns (routing vs grading vs "
        "generation), makes the rewrite loop explicit, and avoids an unbounded tool-chaining policy "
        "for every question. ReAct remains useful as a simpler baseline."
    )

    pdf.section("4. Chunk size and financial tables")
    pdf.body(
        "PDF tables are split by fixed character windows (default chunk_size=2000, overlap=200 in build_rag.py). "
        "Larger chunks keep a full statement block together more often, which helps line items that "
        "only make sense next to their row labels (e.g. Cost of sales - Services vs a quarterly column). "
        "Smaller chunks improve embedding specificity for short queries but risk cutting a table across "
        "chunks, so the retriever may return only the header or only a number. Mitigations in code include "
        "higher fetch_k with MMR, explicit similarity hooks for benchmark anchors (e.g. 25,119; 31,370; 4,540), "
        "and generator instructions to prefer twelve-month columns over three-month figures."
    )

    pdf.section("5. Evaluation")
    pdf.body(
        "The public harness (evaluator.py) runs fixed questions with an LLM judge plus deterministic "
        "checks for key numeric traps (annual vs quarterly, company isolation, trap questions). "
        "EVAL_MODE=GRAPH exercises the LangGraph path; EVAL_MODE=LEGACY exercises the ReAct baseline."
    )

    pdf.output(str(out_path))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Assignment 3 report.pdf")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "report.pdf",
        help="Output PDF path",
    )
    args = p.parse_args()
    build_pdf(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
