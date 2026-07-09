# RAG PDF Question Answering App

This repository contains a simple Retrieval-Augmented Generation (RAG) proof of concept for asking questions about a PDF document.

The app loads a PDF, extracts its text, splits it into chunks, creates embeddings with a sentence-transformer model, stores them in a FAISS index, and uses Gemini to answer questions based on the retrieved context.

## Features

- Reads PDF files and extracts text
- Splits long documents into smaller chunks
- Builds embeddings and a FAISS vector index
- Caches embeddings and index files for faster reuse
- Runs an interactive question-answer loop in the terminal

## Requirements

- Python 3.9+
- A Google Gemini API key

## Setup

1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set your Gemini API key

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

## Usage

Run the app with a PDF file:

```bash
python app.py --pdf "path/to/your.pdf"
```

Optional arguments:

- `--pdf`: path to the PDF file
- `--cache-dir`: directory where cached embeddings and FAISS artifacts are stored

Example:

```bash
<<<<<<< HEAD
python app.py --pdf "17 NISM-Series-XV-Research Analyst Examination Workbook February 2026.pdf" --cache-dir cache
=======
python app.py --pdf "sample.pdf" --cache-dir cache
>>>>>>> 636b23956400b0357cc74575bf1539bd620fa141
```

After startup, type your question and press Enter. Type `exit` to quit.

## Project Structure

- `app.py` - main CLI entrypoint
- `utils.py` - PDF extraction, chunking, FAISS indexing, and search
- `gen_ai.py` - Gemini-based answer generation
- `config.py` - model configuration
- `cache/` - stored embeddings and index artifacts

## Notes

The first run may take some time because the embedding model and FAISS index are built from scratch. Subsequent runs will reuse the cached artifacts when the PDF and its hash match.

