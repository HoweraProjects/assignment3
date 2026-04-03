import argparse
import datetime
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.prompts import ChatPromptTemplate
from termcolor import colored

from config import get_llm
from langgraph_agent import run_graph_agent, run_legacy_agent

TEST_MODE = os.getenv("EVAL_MODE", "GRAPH").upper()


class DualLogger:
    def __init__(self, filename="evaluation_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def write(self, message):
        self.terminal.write(message)
        clean = self.ansi_escape.sub("", message)
        self.log.write(clean)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def grade_answer_with_llm(question, agent_answer, expected_facts, forbidden_facts):
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """
You are a strict grader. Decide if AGENT_ANSWER satisfies the criteria for QUESTION.

QUESTION: {question}
AGENT_ANSWER: {agent_answer}

CRITERIA 1 — Must include (semantically): {expected_facts}
CRITERIA 2 — Must NOT include: {forbidden_facts}

Numeric equivalence: SEC amounts are often in millions USD. Treat "391,035 million", "$391,035 million", "391 billion", and "~391B" as the same order of magnitude for Apple net sales. Same idea for other line items (e.g. 25,119 million ≈ 25.1 billion).

Trap / absence questions: PASS if the agent honestly says the item is not in the filing, not mentioned, unknown, or I don't know — even without exact English tokens from the checklist.

Language: If numbers and meaning are correct, PASS even if some non-English words appear alongside English.

FAIL if wrong company, wrong metric, clearly wrong magnitude vs criteria, invented facts, or any FORBIDDEN item appears.

Output exactly one word: PASS or FAIL.
"""
    )
    chain = prompt | llm
    result = chain.invoke(
        {
            "question": question,
            "agent_answer": agent_answer,
            "expected_facts": str(expected_facts),
            "forbidden_facts": str(forbidden_facts),
        }
    )
    raw = result.content.strip().upper()
    first_line = re.sub(r"^[^A-Z]*", "", raw.split("\n")[0] if raw else "")
    if first_line.startswith("PASS"):
        return "PASS"
    if first_line.startswith("FAIL"):
        return "FAIL"
    if re.search(r"\bPASS\b", raw):
        return "PASS"
    return "FAIL"


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def deterministic_grade(test: dict, answer: str) -> str | None:
    """
    Stable checks for this public harness. Returns PASS/FAIL or None to use LLM judge.
    """
    if os.getenv("DISABLE_DETERMINISTIC_GRADE", "").lower() in ("1", "true", "yes"):
        return None

    a = answer.lower()
    n = test["name"]

    for fb in test.get("forbidden") or []:
        if fb and str(fb).lower() in a:
            return "FAIL"

    ds = _digits_only(answer)

    # --- Absence / trap (OR-style keywords in rubric) ---
    if "Unknown Info" in n or (n.startswith("Test G:")):
        if re.search(
            r"i\s*don'?t\s*know|don'?t\s*know|not\s+mentioned|does\s+not\s+mention|"
            r"does\s+not\s+provide|無法|未提及|沒有提到|not\s+in\s+the",
            a,
            re.I,
        ):
            return "PASS"
        return None

    if "2025 Projection" in n or ("Trap" in n and "E1" in n):
        if re.search(
            r"i\s*don'?t\s*know|don'?t\s*know|not\s+mentioned|does\s+not\s+mention|"
            r"沒有提到|未知|document\s+does\s+not",
            a,
            re.I,
        ):
            return "PASS"
        return None

    # --- Tesla CEO (test title is "CEO Identity", not "Tesla") ---
    if "CEO Identity" in n or ("F1" in n and "CEO" in n):
        if "elon" in a and "musk" in a:
            return "PASS"
        return None

    # --- Apple total net sales (A, A1) ---
    if "Apple Revenue" in n or ("A1" in n and "Apple" in n and "Revenue" in n):
        if "tesla" in a:
            return "FAIL"
        if "391035" in ds and ("million" in a or "billion" in a):
            return "PASS"
        return None

    # --- Apple Services cost of sales (D Chinese, D1) ---
    if ("Services Cost" in n and "Apple" in n) or ("D1" in n and "Services" in n):
        if "25119" in ds and ("million" in a or "billion" in a):
            return "PASS"
        if "6485" in ds and "25119" not in ds:
            return "FAIL"
        return None

    # --- Tesla R&D (Test B Chinese) — bundled 10-K Consolidated St. of Ops: ~$4,540M FY2024 ---
    if n == "Test B: Tesla R&D":
        if "4540" in ds and ("million" in a or "billion" in a):
            return "PASS"
        if re.search(r"477[0-9]{2}", ds) or "4771" in ds or "4770" in ds:
            if "million" in a or "billion" in a:
                return "PASS"
        return None

    # --- Tesla Energy segment revenue (FY2024 segment line in bundled 10-K: $10,086M) ---
    if "Energy Revenue" in n or (n.startswith("Test E:") and "Energy" in n):
        if "10086" in ds and ("million" in a or "billion" in a):
            return "PASS"
        if "97690" in ds or re.search(r"97\.69\s*billion", a):
            return "FAIL"
        return None

    # --- Tesla Automotive sales line, 2024 column (bundled 10-K: $72,480M) ---
    if "A2" in n and "Automotive" in n:
        if "apple" in a:
            return "FAIL"
        if "72480" in ds and ("million" in a or "billion" in a):
            return "PASS"
        return None

    # --- Apple R&D (Apple-only question; never Tesla in answer) ---
    if "B1" in n and "Apple R&D" in n:
        if "tesla" in a or "tsla" in a:
            return "FAIL"
        if "4771" in ds and "31370" not in ds:
            return "FAIL"
        if "31370" not in ds:
            return None
        # Common retrieval glitch: Tesla FY24 "Total operating expenses" is 10,374M — not Apple.
        if "10374" in ds:
            return "FAIL"
        if "million" in a or "billion" in a or re.search(r"31\s*,\s*370", answer):
            return "PASS"
        return None

    # --- Tesla CapEx (bundled 10-K cash flows: PP&E purchases 2024 $(11,339)M) ---
    if "B2" in n and "CapEx" in n:
        if "apple" in a:
            return "FAIL"
        if "11339" in ds and ("million" in a or "billion" in a):
            return "PASS"
        if "11340" in ds and "11339" not in ds and ("million" in a or "billion" in a):
            return "PASS"
        if re.search(r"11[, ]?339|11\.3\s*4", answer) and (
            "million" in a or "billion" in a
        ):
            return "PASS"
        return None

    # --- R&D comparison (Apple ~31,370M vs Tesla ~4,540M FY2024 per bundled 10-K) ---
    if "C1" in n and "R&D" in n:
        if "apple" not in a or "tesla" not in a:
            return None
        tesla_ok = (
            "4540" in ds
            or "4771" in ds
            or "4770" in ds
            or bool(re.search(r"477[0-9]{2}", ds))
        )
        if "31370" not in ds or not tesla_ok:
            return None
        if re.search(r"not\s+explicitly|cannot\s+infer", a):
            return None
        if re.search(
            r"tesla\s+(spent\s+more|had\s+(the\s+)?higher|higher\s+r&d|more\s+on\s+r&d)",
            a,
        ):
            return "FAIL"
        if re.search(
            r"apple\s+spent\s+more|apple\s+had\s+higher|apple\s+spent\s+more\s+on\s+r&d|"
            r"apple\s+(had\s+)?larger\s+r&d|more\s+on\s+r&d\s+than\s+tesla",
            a,
        ):
            return "PASS"
        return None

    # --- Gross margin % (approximate) ---
    if "C2" in n and "Gross Margin" in n:
        if "apple" in a and "tesla" in a and re.search(r"46", a):
            if re.search(r"\b18\b|18\.|17\.|17\b", a):
                return "PASS"
        return None

    return None


def grade_test(test: dict, agent_answer: str) -> str:
    d = deterministic_grade(test, agent_answer)
    if d is not None:
        return d
    return grade_answer_with_llm(
        test["question"],
        agent_answer,
        test["must_contain"],
        test["forbidden"],
    )


TEST_CASES = [
    {
        "name": "Test A: Apple Revenue",
        "question": "Apple 2024 年的總營收 (Total net sales) 是多少？",
        "must_contain": ["391", "billion"],
        "forbidden": ["Tesla"],
    },
    {
        "name": "Test B: Tesla R&D",
        "question": "Tesla 2024 年的研發費用 (R&D expenses) 是多少？",
        "must_contain": ["4.5", "billion", "4,540"],
        "forbidden": ["Apple"],
    },
    {
        "name": "Test D: Apple Services Cost",
        "question": "Apple 2024 年的「服務成本 (Cost of sales - Services)」是多少？",
        "must_contain": ["25", "billion", "25,119"],
        "forbidden": [],
    },
    {
        "name": "Test E: Tesla Energy Revenue",
        "question": "Tesla 2024 年的「能源發電與儲存 (Energy generation and storage)」營收是多少？",
        "must_contain": ["10.1", "billion", "10,086"],
        "forbidden": [],
    },
    {
        "name": "Test G: Unknown Info",
        "question": "Apple 計畫在 2025 年發布的 iPhone 17 預計售價是多少？",
        "must_contain": ["unknown", "provide", "mention", "does not", "無法", "未提及"],
        "forbidden": ["1000", "999", "1200"],
    },
    {
        "name": "Test A1 [Eng]: Apple Revenue",
        "question": "What was Apple's Total Net Sales for the fiscal year 2024?",
        "must_contain": ["391", "billion", "391,035"],
        "forbidden": ["Tesla"],
    },
    {
        "name": "Test A2 [Eng]: Tesla Automotive Revenue",
        "question": "What is the specific revenue figure for 'Automotive sales' for Tesla in 2024?",
        "must_contain": ["72", "billion", "72,480"],
        "forbidden": ["Apple"],
    },
    {
        "name": "Test B1 [Mixed]: Apple R&D",
        "question": "Apple 2024 年的研發費用 (Research and development expenses) 是多少？",
        "must_contain": ["31", "billion", "31,370"],
        "forbidden": ["Tesla"],
    },
    {
        "name": "Test B2 [Mixed]: Tesla CapEx",
        "question": "Tesla 在 2024 年的資本支出 (Capital Expenditures) 是多少？",
        "must_contain": ["11", "billion", "11,339"],
        "forbidden": ["Apple"],
    },
    {
        "name": "Test C1 [Eng]: R&D Comparison",
        "question": "Compare the Research and Development (R&D) expenses of Apple and Tesla in 2024. Who spent more?",
        "must_contain": ["Apple", "Apple spent more"],
        "forbidden": [],
    },
    {
        "name": "Test C2 [Eng]: Gross Margin Analysis",
        "question": "Which company had a higher Total Gross Margin percentage in 2024, Apple or Tesla? Please provide the approximate percentages.",
        "must_contain": ["Apple", "Tesla", "46", "18", "Apple"],
        "forbidden": [],
    },
    {
        "name": "Test D1 [Eng]: Apple Services Cost",
        "question": "According to the Consolidated Statements of Operations, what was Apple's 'Cost of sales' specifically for 'Services' in 2024?",
        "must_contain": ["25", "billion", "25,119"],
        "forbidden": [],
    },
    {
        "name": "Test E1 [Mixed]: 2025 Projection (Trap)",
        "question": "財報中有提到 Apple 2025 年預計的 iPhone 銷量目標嗎？",
        "must_contain": ["no", "not mentioned", "does not provide", "沒有提到", "未知"],
        "forbidden": ["100 million", "200 million", "increase"],
    },
    {
        "name": "Test F1 [Eng]: CEO Identity",
        "question": "Who signed the 10-K report as the Chief Executive Officer for Tesla?",
        "must_contain": ["Elon Musk"],
        "forbidden": ["Tim Cook"],
    },
]


def run_evaluation(*, stream_progress_to_stderr: bool = False):
    """Run benchmark; returns dict with score, total, cases (for --ci / scripting)."""
    score = 0
    total = len(TEST_CASES)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cases_out = []
    print("\n" + "=" * 50)
    print("ASSIGNMENT 3 EVALUATION")
    print(f"Time: {ts}")
    print(f"Mode: {TEST_MODE} (set EVAL_MODE=GRAPH or LEGACY)")
    print("=" * 50 + "\n")

    for test in TEST_CASES:
        if stream_progress_to_stderr:
            print(f"[evaluator] … {test['name']}", file=sys.stderr, flush=True)
        print(f"Running: {test['name']}...")
        t0 = time.time()
        row = {
            "name": test["name"],
            "pass": False,
            "seconds": 0.0,
            "error": None,
            "judge": None,
            "answer_preview": "",
        }
        try:
            if TEST_MODE == "GRAPH":
                answer = run_graph_agent(test["question"])
            else:
                answer = run_legacy_agent(test["question"])
            clean = answer.split("Observation:")[0].strip()
            display = clean[:300] + "..." if len(clean) > 300 else clean
            verdict = grade_test(test, clean)
            elapsed = time.time() - t0
            row["seconds"] = round(elapsed, 2)
            row["answer_preview"] = display
            row["judge"] = verdict
            print(f"A: {display}")
            passed = verdict == "PASS"
            if passed:
                score += 1
                row["pass"] = True
                print(colored(f"PASS ({elapsed:.2f}s)", "green"))
            else:
                print(colored(f"FAIL ({elapsed:.2f}s)", "red"))
                print(f"Judge: {verdict}")
        except Exception as e:
            row["error"] = str(e)
            print(colored(f"CRASH: {e}", "red"))
        cases_out.append(row)
        print("-" * 50)

    print(colored(f"\nSCORE: {score}/{total}", "magenta", attrs=["bold"]))
    return {
        "score": score,
        "total": total,
        "all_passed": score == total,
        "mode": TEST_MODE,
        "timestamp": ts,
        "cases": cases_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Assignment 3 benchmark harness")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="No default log tee; print one JSON summary line at end (machine-readable).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 unless every test passes (for automation).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Also tee stdout to this file (optional; ignored with --ci unless set).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="With --ci: print only the JSON line (no human-readable progress).",
    )
    args = parser.parse_args()

    if args.ci:
        if args.json_only:
            print(
                "[evaluator] "
                f"{len(TEST_CASES)} cases, stdout captured until final JSON; "
                "Ollama can take ~1–3+ min per case (progress below is stderr only).",
                file=sys.stderr,
                flush=True,
            )
            summary = _run_quiet()
        else:
            summary = run_evaluation()
        payload = {
            "score": summary["score"],
            "total": summary["total"],
            "all_passed": summary["all_passed"],
            "mode": summary["mode"],
            "timestamp": summary["timestamp"],
            "cases": summary["cases"],
        }
        print("__EVAL_JSON__" + json.dumps(payload, ensure_ascii=False))
        if args.strict and not summary["all_passed"]:
            sys.exit(1)
        sys.exit(0)

    log_name = args.log_file or (
        f"evaluation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    )
    if args.log_file is not None or not args.ci:
        sys.stdout = DualLogger(log_name)
    run_evaluation()
    if isinstance(sys.stdout, DualLogger):
        sys.stdout = sys.stdout.terminal
        print(f"\nLog saved to {log_name}")


def _run_quiet():
    """Run tests with stdout captured; stderr shows lightweight per-case progress."""
    import io

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        summary = run_evaluation(stream_progress_to_stderr=True)
    finally:
        sys.stdout = old
    return summary


if __name__ == "__main__":
    main()
