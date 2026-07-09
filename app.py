import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

from utils import (
    artifacts_exist,
    extract_text_from_pdf,
    load_rag_artifacts,
    save_rag_artifacts,
    search,
    split_into_chunks,
    build_faiss_index,
)

from gen_ai import ask_gemini

parser = argparse.ArgumentParser(description="PDF RAG app with cached embeddings")
parser.add_argument(
    "--pdf",
    default="17 NISM-Series-XV-Research Analyst Examination Workbook February 2026.pdf",
    help="Path to the PDF file",
)
parser.add_argument(
    "--cache-dir",
    default="cache",
    help="Directory to store cached artifacts",
)
args = parser.parse_args()

pdf_file = Path(args.pdf)
cache_dir = Path(args.cache_dir)

if not pdf_file.exists():
    raise FileNotFoundError(f"PDF file not found: {pdf_file}")

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

if artifacts_exist(pdf_file, cache_dir):
    print("Cached artifacts found. Loading from disk...")
    chunks, embeddings, index = load_rag_artifacts(pdf_file, cache_dir)
else:
    print("Reading PDF...")
    text = extract_text_from_pdf(str(pdf_file))

    print(f"Characters extracted: {len(text):,}")
    print("Creating chunks...")

    chunks = split_into_chunks(
        text,
        chunk_size=300,
        overlap=100
    )

    print(f"Total chunks: {len(chunks)}")

    print("Building FAISS index...")
    index, embeddings = build_faiss_index(
        chunks,
        model
    )

    print(f"FAISS vectors stored: {index.ntotal}")
    save_rag_artifacts(pdf_file, chunks, embeddings, index, cache_dir)

# ----------------------------
# Query Loop
# ----------------------------
while True:

    query = input("\nAsk Question (or 'exit'): ")

    if query.lower() == "exit":
        break

    result = search(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=3
    )



    answer = ask_gemini(query, result["context"])
    print("\nAnswer from Gemini:")
    print("=" * 150)
    print(answer)
    print("=" * 150) 