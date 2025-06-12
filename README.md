# AI Document Summarizer

A document summarization tool that uses RAG (Retrieval-Augmented Generation) to generate summaries of PDF and text documents. Supports OpenAI, Azure OpenAI, and custom embedding endpoints.

## Features

- **Multi-format**: PDF and text file support
- **RAG-based**: Context-aware summarization via chunking, embedding, and retrieval
- **Flexible providers**: OpenAI, Azure OpenAI (chat), and custom embedding endpoints
- **Monitoring**: Optional LangFuse integration
- **Export**: Download summaries as text files

## Quick Start

```bash
# Setup
chmod +x setup.sh && ./setup.sh

# Configure
cp config.example .env
# Edit .env with your API keys

# Run
source .venv/bin/activate
streamlit run app.py
```

Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example .env
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Configuration

Copy `config.example` to `.env` and set the relevant variables.

### Option 1: OpenAI

```env
OPENAI_API_KEY=your_key
OPENAI_API_TYPE=openai

# Optional
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### Option 2: Azure OpenAI + Custom Embeddings

```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt-deployment
OPENAI_API_TYPE=azure

# Custom embedding endpoint (used instead of Azure embeddings)
CUSTOM_EMBEDDING_URL=http://your-host:port/encoder/bi-encoder-embed
CUSTOM_EMBEDDING_MODEL=your-embedding-model-name
```

### Optional: LangFuse Monitoring

```env
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

## How RAG Works Here

```
Upload -> Split into chunks -> Embed each chunk -> Store in ChromaDB
                                                        |
Query -> Embed query -> Find top 4 similar chunks -> Stuff into prompt -> LLM -> Summary
```

1. **Extract**: Pull text from PDF (via PyPDF) or text file
2. **Chunk**: Split into ~1000-char pieces with 200-char overlap
3. **Embed**: Convert chunks to 768-dim vectors (your-embedding-model-name or OpenAI embeddings)
4. **Store**: Index vectors in ChromaDB
5. **Retrieve**: Find the 4 most relevant chunks for the summary query
6. **Generate**: LLM produces a summary grounded in those chunks

## Project Structure

```
ai-app/
├── app.py              # Streamlit app + DocumentSummarizer class
├── requirements.txt    # Python dependencies
├── setup.sh            # Automated setup script
├── config.example      # .env template
├── sample_document.txt # Example document for testing
├── .gitignore          # Git ignore rules
├── .env                # Your API keys (not committed)
└── README.md           # Project documentation
```

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI |
| langchain / langchain-openai / langchain-community | RAG orchestration |
| chromadb | Vector database |
| pypdf | PDF text extraction |
| requests | Custom embedding endpoint calls |
| langfuse | Optional observability |

## Troubleshooting

**API key errors** -- check `.env` exists and has the correct values.

**ChromaDB write errors** -- delete the `chroma_db/` directory and retry.

**Custom embedding timeout** -- ensure the embedding server at `CUSTOM_EMBEDDING_URL` is running.

**LangFuse import errors** -- LangFuse is optional; the app works without it.
