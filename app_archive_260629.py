import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Read PDF
# --------------------------------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# 2. Split text into chunks
# --------------------------------------------------
def split_into_chunks(text, chunk_size=300, overlap=100):
    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if len(chunk.strip()) > 0:
            chunks.append(chunk)

    return chunks


# --------------------------------------------------
# 3. Build FAISS Index
# --------------------------------------------------
def build_faiss_index(chunks, model):
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings


# --------------------------------------------------
# 4. Search
# --------------------------------------------------
def search(query, model, index, chunks, top_k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_indices = indices[0]

    # preserve original document order
    ordered_indices = sorted(retrieved_indices)

    retrieved_chunks = [
        chunks[i]
        for i in ordered_indices
    ]

    context = "\n\n".join(retrieved_chunks)

    return {
        "distances": distances[0],
        "indices": retrieved_indices,
        "context": context,
        "chunks": retrieved_chunks
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    pdf_file = "17 NISM-Series-XV-Research Analyst Examination Workbook February 2026.pdf"

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Reading PDF...")
    text = extract_text_from_pdf(pdf_file)

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

        print("\nRetrieved Indices:")
        print(result["indices"])

        print("\nDistances:")
        print(result["distances"])

        print("\nRetrieved Context:")
        print("=" * 80)
        print(result["context"])
        print("=" * 80)