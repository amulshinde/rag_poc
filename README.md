# Simple RAG using FAISS and Google Gemini

A lightweight Retrieval-Augmented Generation (RAG) application built with Python that allows users to ask questions from PDF documents.

The project extracts text from a PDF, splits it into chunks, generates embeddings using Sentence Transformers, stores them in a FAISS vector database, retrieves the most relevant chunks for a query, and uses Google Gemini to generate an accurate response.

---

## Features

- Extract text from PDF files
- Split text into manageable chunks
- Generate embeddings using `all-MiniLM-L6-v2`
- Store embeddings in a FAISS vector database
- Save and load FAISS index locally
- Retrieve relevant document chunks using semantic search
- Generate answers using Google Gemini
- API key stored securely using `.env`

---

## Tech Stack

- Python
- FAISS
- Sentence Transformers
- Google Gemini API
- PyPDF2
- python-dotenv

---

## Project Workflow

```text
                PDF
                 │
                 ▼
        Extract Text
                 │
                 ▼
         Split into Chunks
                 │
                 ▼
Generate Embeddings
(all-MiniLM-L6-v2)
                 │
                 ▼
      Store in FAISS Index
                 │
─────────────────┼─────────────────
                 │
            User Query
                 │
                 ▼
Generate Query Embedding
                 │
                 ▼
 Search FAISS Index
                 │
                 ▼
 Retrieve Top-K Chunks
                 │
                 ▼
 Send Context + Query
      to Gemini API
                 │
                 ▼
      Generated Answer
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```text
GOOGLE_API_KEY=YOUR_API_KEY
```

---

## Run

Create the FAISS index

```bash
python create_index.py
```

Ask questions

```bash
python rag.py
```

---

## Folder Structure

```
project/
│
├── data/
│   └── sample.pdf
│
├── faiss_index/
│
├── config.py
├── create_index.py
├── rag.py
├── gen_ai.py
├── utils.py
├── requirements.txt
├── .env
└── README.md
```

---

## Model Used

Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

LLM

```
Google Gemini
```

Vector Database

```
FAISS
```

---

## Future Improvements

- Multiple PDF support
- Metadata filtering
- Hybrid Search (BM25 + FAISS)
- Streamlit interface
- Conversation memory
- Reranking
- Source citations
- Docker support

---

## License

MIT License
