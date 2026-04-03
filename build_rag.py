import os
import re

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from termcolor import colored

from config import DATA_FOLDER, DB_FOLDER, FILES, get_embeddings

# Assignment: default chunk_size=2000; tune CHUNK_SIZE / CHUNK_OVERLAP for experiments.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_vector_dbs():
    embeddings = get_embeddings()

    os.makedirs(DATA_FOLDER, exist_ok=True)

    all_files = dict(FILES)
    known_names = set(FILES.values())
    for f in os.listdir(DATA_FOLDER):
        if not f.endswith(".pdf"):
            continue
        if f in known_names:
            continue
        key = f.split(".")[0].lower()
        if key not in all_files:
            all_files[key] = f
            print(colored(f"Found extra PDF: {f} (key '{key}')", "green"))

    for key, filename in all_files.items():
        persist_dir = os.path.join(DB_FOLDER, key)
        file_path = os.path.join(DATA_FOLDER, filename)

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print(
                colored(
                    f"DB for '{key}' exists at {persist_dir}; skip (delete folder to rebuild).",
                    "yellow",
                )
            )
            continue

        if not os.path.exists(file_path):
            print(colored(f"Missing source file: {file_path}", "red"))
            continue

        print(colored(f"Building index: {key} (chunk_size={CHUNK_SIZE})...", "cyan"))

        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        print(f"  Loaded {len(docs)} pages.")

        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        splits = splitter.split_documents(docs)
        print(f"  Split into {len(splits)} chunks.")

        os.makedirs(persist_dir, exist_ok=True)
        print("  Embedding and storing...")
        Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
        print(colored(f"Done: {key}", "green"))


if __name__ == "__main__":
    build_vector_dbs()
