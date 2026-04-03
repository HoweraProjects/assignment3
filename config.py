import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from termcolor import colored

load_dotenv(override=True)

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
DB_ROOT = os.path.join(os.path.dirname(__file__), "chroma_db")

FILES = {
    "apple": "FY24_Q4_Consolidated_Financial_Statements.pdf",
    "tesla": "tsla-20241231-gen.pdf",
}

# Separate DB per embedding tag so you can compare models (report benchmark).
EMBEDDING_TAG = os.getenv("EMBEDDING_TAG", "default").strip() or "default"
DB_FOLDER = os.path.join(DB_ROOT, EMBEDDING_TAG)

# Swap via env to compare embeddings (assignment report).
LOCAL_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


def get_embeddings():
    print(
        colored(
            f"Loading embeddings [{EMBEDDING_TAG}]: {LOCAL_EMBEDDING_MODEL}...",
            "cyan",
        )
    )
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)


def get_llm(temperature=0):
    """
    LLM factory. Set LLM_PROVIDER=ollama (default) and OLLAMA_MODEL for local runs.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(colored("Warning: GOOGLE_API_KEY not set.", "red"))
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True,
            max_output_tokens=2048,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(colored("Warning: OPENAI_API_KEY not set.", "red"))
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=api_key,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
