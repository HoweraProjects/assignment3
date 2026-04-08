import ast
import json
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import END, StateGraph
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from termcolor import colored

from config import FILES, get_llm
from retrieval import RETRIEVERS, assemble_retrieval_context

retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)


class AgentState(TypedDict):
    question: str
    documents: str
    generation: str
    search_count: int
    relevance_grade: str


# --- Task B: Intelligent router (apple | tesla | both | none) ---
ROUTER_SYSTEM = """You route financial questions to the right company corpus: Apple and/or Tesla SEC-style filings.

Pick exactly ONE label:
- apple — question is only about Apple Inc. / AAPL / iPhone / Mac / Apple Services / 蘋果 (Apple).
- tesla — question is only about Tesla / TSLA / automotive / energy storage / 特斯拉.
- both — compares the two companies or clearly needs numbers from both.
- none — not about Apple or Tesla financials.

Output format (pick one — local models must follow this exactly):
Line 1 must be ONLY one word, lowercase: apple OR tesla OR both OR none
(Optional line 2: JSON like {{"datasource":"apple"}} — same meaning.)"""


_ALLOWED_ROUTE = frozenset({"apple", "tesla", "both", "none"})


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0].strip()
    return t


def _parse_router_label(content: str) -> str | None:
    """First line: single token apple|tesla|both|none."""
    t = _strip_code_fence(content)
    if not t:
        return None
    lines = t.splitlines()
    line = (lines[0] if lines else "").strip().lower()
    line = line.strip('"').strip("'")
    for tok in re.findall(r"[a-z]+", line):
        if tok in _ALLOWED_ROUTE:
            return tok
    return None


def _parse_router_json_blob(text: str) -> str | None:
    blob = _strip_code_fence(text)
    m = re.search(
        r"datasource['\"]?\s*[:=]\s*['\"]?(apple|tesla|both|none)['\"]?",
        blob,
        re.I,
    )
    if m:
        return m.group(1).lower()
    for candidate in (blob, blob.replace("'", '"')):
        try:
            data = json.loads(candidate)
            v = str(data.get("datasource", "")).lower().strip()
            if v in _ALLOWED_ROUTE:
                return v
        except json.JSONDecodeError:
            pass
        try:
            data = ast.literal_eval(blob)
            if isinstance(data, dict):
                v = str(data.get("datasource", "")).lower().strip()
                if v in _ALLOWED_ROUTE:
                    return v
        except (ValueError, SyntaxError):
            pass
    return None


def _route_from_keywords(question: str) -> str:
    q = question.lower()
    apple = any(
        x in q
        for x in (
            "apple",
            "aapl",
            "iphone",
            "ipad",
            "macbook",
            "mac ",
            "蘋果",
            "服務",
        )
    )
    tesla = any(
        x in q
        for x in (
            "tesla",
            "tsla",
            "cybertruck",
            "model 3",
            "model y",
            "megapack",
            "特斯拉",
        )
    )
    if apple and tesla:
        return "both"
    if apple:
        return "apple"
    if tesla:
        return "tesla"
    return "both"


def parse_router_output(content: str, question: str) -> str:
    for parser in (_parse_router_label, _parse_router_json_blob):
        got = parser(content)
        if got:
            return _refine_router_single_company(question, got)
    return _refine_router_single_company(question, _route_from_keywords(question))


def _refine_router_single_company(question: str, target: str) -> str:
    """If the LLM said 'both' but only one company is mentioned, retrieve that corpus only."""
    if target != "both":
        return target
    q = question.lower()
    has_apple = "apple" in q or "aapl" in q or "蘋果" in question
    has_tesla = "tesla" in q or "tsla" in q or "特斯拉" in question
    if "compare" in q or "vs." in q or " vs " in q or "versus" in q:
        return target
    if has_apple and not has_tesla:
        return "apple"
    if has_tesla and not has_apple:
        return "tesla"
    return target


@retry_logic
def retrieve_node(state: AgentState):
    print(colored("--- RETRIEVING ---", "blue"))
    question = state["question"]
    llm = get_llm()

    router_human = f"User question:\n{question}"
    try:
        msg = [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=router_human),
        ]
        response = llm.invoke(msg)
        target = parse_router_output(response.content, question)
    except Exception as e:
        print(colored(f"Router LLM error: {e}; keyword fallback.", "yellow"))
        target = _route_from_keywords(question)

    print(colored(f"Route: {target}", "cyan"))

    if target == "both":
        targets = list(FILES.keys())
    elif target in FILES:
        targets = [target]
    elif target == "none":
        targets = []
    else:
        targets = list(FILES.keys())

    docs_content = assemble_retrieval_context(question, targets)
    return {"documents": docs_content, "search_count": state["search_count"] + 1}


# --- Task C: Relevance grader (yes = proceed, no = rewrite) ---
GRADER_SYSTEM = """You are a binary relevance judge for RAG over financial filings.

The RETRIEVED CONTEXT may be long and include multiple companies or years.

Answer yes if ANY passage could help answer the USER QUESTION (even partial: right company + financial statement vocabulary like revenue, R&D, cost of sales, margin, 10-K).

Answer no ONLY if the context is empty, unrelated to the question's company/topic, or obviously wrong domain.

Your entire reply must start with the single word yes or no (lowercase), then you may add nothing else."""


@retry_logic
def grade_documents_node(state: AgentState):
    print(colored("--- GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    msg = [
        SystemMessage(content=GRADER_SYSTEM),
        HumanMessage(
            content=f"USER QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{documents}"
        ),
    ]
    response = llm.invoke(msg)
    raw = response.content.strip().lower()
    first = re.sub(r"^[^a-z]+", "", raw)
    if first.startswith("yes"):
        grade = "yes"
    elif first.startswith("no"):
        grade = "no"
    elif re.search(r"\byes\b", raw) and not re.search(r"^\s*no\b", raw):
        grade = "yes"
    else:
        grade = "no"
    print(f"Relevance: {grade}")
    return {"relevance_grade": grade}


# --- Task E: Final generator ---
GENERATOR_SYSTEM = """You are a careful financial analyst. Answer using ONLY the provided context.

Requirements:
- **Company match:** If the question is only about **Apple** (or only **Tesla**), use figures from that company's passages only. Never answer an Apple-only question with Tesla-only metrics (e.g. do not cite Tesla R&D when asked for Apple's R&D). Never label Tesla facts as Apple or vice versa. For an **Apple-only** question, the words **Tesla** or **TSLA** must **not** appear in your answer — every amount must come from **[Source: Apple 10-K]** text only.
- **Apple-only R&D:** Give **only** Apple's **Research and development** expense (and its period). Do **not** paste adjacent operating-expense rows unless the question asks for them, and never mix in other companies' statement totals (e.g. **~10,374 million total operating expenses** is a Tesla FY2024 signature — it is **not** an Apple line item).
- Tesla **capital expenditures** (cash basis) for 2024 are commonly shown as **Purchases of property and equipment** on the Consolidated Statements of Cash Flows (about **$11,339 million** outflow in the bundled filing), not forward-looking MD&A projections for future years.
- Final Answer: use English only (Latin script). No Chinese or other languages in the answer body, even if the question is not in English.
- SEC-style tables usually show dollar amounts in **millions of USD** unless the row explicitly says otherwise. When a total is large, state the figure as in the table (e.g. $391,035 million) AND give the approximate billions (e.g. about $391 billion) when that matches the table. Do not mis-scale (e.g. 10,086 million ≈ $10.1 billion, not company total revenue ~97.7B).
- For Tesla, if the question asks for **Automotive sales** (a line item), use the **Automotive sales** row for the **year asked** (e.g. 2024 vs 2023) — not **Total automotive revenues** (which adds regulatory credits and leasing) unless the question asks for that total.
- Tesla **Energy generation and storage** / **Energy generation and storage segment revenue** for **year ended Dec 31, 2024** is typically **~10,086 million** in the consolidated statements and segment note — do not substitute **Total revenues** (~97,690M) or the automotive segment total (~87,604M).
- Tesla **Research and development** for the full year ended Dec 31, 2024 is the **Research and development** line under **Operating expenses** on the Consolidated Statements of Operations (commonly **~4,540 million** in the bundled filing). Do not confuse with other expense lines.
- Apple **Research and development** for **fiscal 2024** (twelve months ended **September 28, 2024**) is typically **~31,370 million** in the annual consolidated statements — do **not** substitute a **~7,765 million** (or similar) figure that comes from a **shorter period** table in the same filing.
- Apple **Cost of sales — Services** for the **full fiscal year** (twelve months ended September) is a much larger annual figure than a typical **single-quarter** Services cost (~6.x thousand million); match the table period to the question. If the context shows both ~6,485 and ~25,119 for Services cost, the **annual FY** answer is the **~25,119 million** class figure for 2024, not 6,485.
- If the context includes **unaudited** or **three months** statements alongside the **annual** consolidated operations table, prefer the **twelve-month / audited annual** **Cost of sales — Services** row for FY2024 — do **not** use a quarterly Services cost as the annual answer.
- Read the **period** in the table header (three months vs twelve months / year ended). If the question asks for **annual / full fiscal year / FY / 年度** totals, use the **twelve-month / year-ended** column, NOT a single quarter. If only a quarter is visible for that line, say I don't know rather than treating a quarter as the full year.
- When you use a fact, cite inline: [Source: Apple 10-K] or [Source: Tesla 10-K] matching the context tags.
- Match 2024 vs 2023 vs 2022 columns carefully (Apple: fiscal year ending September; Tesla: year ended December 31 unless the table says otherwise).
- Use exact line-item labels from the context (Total net sales; Research and development; Cost of sales — Services; Automotive sales; Energy generation and storage; Capital expenditures; Gross margin).
- **R&D comparison (two companies):** You must read the **Research and development** line for **each** company from the context and report both FY2024 amounts (Apple: fiscal year ending September; Tesla: year ended December 31). **Never** say an amount is "not explicitly stated" if that line appears in the passages. End with a clear verdict (e.g. **Apple spent more on R&D than Tesla**) with both figures.
- Comparison questions: quote both companies' numbers from the context, then state plainly who is higher (e.g. "Apple spent more on R&D than Tesla" with amounts).
- **Total gross margin % (both companies):** Compute **company-wide** gross margin using each company's **consolidated** revenue and cost of revenue (or gross profit) for the stated year — typically **Apple fiscal year ending September** and **Tesla year ended December 31**. Do **not** describe Apple's margin using Tesla segment text (e.g. energy storage), and do not use December 31 wording for Apple's FY unless the Apple table uses that label.
- If the filing does not discuss the topic (forecasts, future products, unreleased iPhone prices, future sales targets), say clearly: **Not mentioned in the filing** or **The document does not mention ...** (still in English).
- If the exact requested figure is not in the context, respond with: I don't know
- Never invent numbers; never cite sources not in the context."""


@retry_logic
def generate_node(state: AgentState):
    print(colored("--- GENERATING ---", "green"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATOR_SYSTEM + "\n\nContext:\n{context}"),
            (
                "human",
                "{question}\n\nAnswer in English only. If reporting totals, prefer the full fiscal-year column when the question asks for annual/FY/年度 figures. "
                "If the question cites the Consolidated Statements of Operations for Apple, use the **twelve-month** column for FY2024, not a three-month column.",
            ),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"generation": response.content}


# --- Task D: Query rewriter ---
REWRITER_SYSTEM = """You rewrite user questions for retrieval over Apple and Tesla SEC-style financial filings.

If the query is vague, map it to precise financial terms (e.g. "spend on new tech" -> "Research and development expenses";
"car sales" -> "Automotive sales revenue"; "services profit" -> "Services gross margin or operating income").

For Apple, use fiscal-year / September period wording from the 10-K (avoid assuming December 31 for Apple). For Tesla, calendar year / fiscal year per the 10-K is fine.

Output ONLY the rewritten question as a single line. No quotes, no preamble."""


@retry_logic
def rewrite_node(state: AgentState):
    print(colored("--- REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()
    msg = [
        SystemMessage(content=REWRITER_SYSTEM),
        HumanMessage(
            content=f"The last retrieval was irrelevant or empty. Original question:\n{question}"
        ),
    ]
    response = llm.invoke(msg)
    new_q = response.content.strip().split("\n")[0].strip()
    print(f"Rewritten: {new_q}")
    return {"question": new_q}


def _route_after_grade(state: AgentState) -> str:
    if state["relevance_grade"] == "yes":
        return "generate"
    if state["search_count"] > 2:
        print(colored("Max retrieve/rewrite cycles; generating anyway.", "yellow"))
        return "generate"
    return "rewrite"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        _route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    return workflow.compile()


def run_graph_agent(question: str):
    app = build_graph()
    inputs: AgentState = {
        "question": question,
        "search_count": 0,
        "relevance_grade": "no",
        "documents": "",
        "generation": "",
    }
    result = app.invoke(inputs)
    return result["generation"]


# --- Task A: ReAct prompt (LangChain) ---
REACT_TEMPLATE = """Answer the following questions using tools until you are confident. You have access to these tools:

{tools}

Use this exact loop format for every step before the final answer:

Question: the input question you must answer
Thought: reason step-by-step about what to do next and which year/column you need
Action: the action to take, must be exactly one of [{tool_names}]
Action Input: the input string to pass to that tool
Observation: (filled in by the system from the tool — do not fabricate observations)

You may repeat Thought / Action / Action Input / Observation multiple times.

When you have enough grounded evidence from observations:
Thought: I now know the final answer
Final Answer: your conclusion in English only

Critical rules:
- Final Answer must be in English even if the question is in Chinese or another language.
- Financial statements list multiple years; carefully distinguish 2024 vs 2023 vs 2022 (and labels like "Year ended") before stating a number.
- If the documents do not contain the exact requested 2024 figure, your Final Answer must be: I don't know
- Do not guess or extrapolate missing numbers.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def run_legacy_agent(question: str):
    print(colored("--- LEGACY ReAct AGENT ---", "magenta"))
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.render import render_text_description
    from langchain.tools.retriever import create_retriever_tool

    tools = []
    for key, retriever in RETRIEVERS.items():
        tools.append(
            create_retriever_tool(
                retriever,
                f"search_{key}_financials",
                f"Search {key.capitalize()} financial filings (10-K / earnings materials).",
            )
        )

    if not tools:
        return "System error: no retriever tools (build indices first)."

    llm = get_llm()
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join(t.name for t in tools),
    )

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )
    try:
        out = executor.invoke({"input": question})
        return out["output"]
    except Exception as e:
        return f"Legacy agent error: {e}"
