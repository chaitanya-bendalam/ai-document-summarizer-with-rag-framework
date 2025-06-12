import streamlit as st
import os
import io
import shutil
from typing import List

import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_DIR = "./chroma_db"
SUMMARY_PROMPT = PromptTemplate(
    template="""Based on the following document excerpts, provide a comprehensive summary that covers:
1. Main topics and key points
2. Important details and findings
3. Conclusions or recommendations (if any)

Document excerpts:
{context}

Question: {question}

Please provide a well-structured summary:""",
    input_variables=["context", "question"],
)

langfuse_handler = None
try:
    from langfuse.langchain import CallbackHandler
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        langfuse_handler = CallbackHandler()
except (ImportError, Exception):
    pass


class CustomEmbeddings(Embeddings):
    def __init__(self, url: str, model_name: str, max_context_length: int = 512):
        self.url = url
        self.model_name = model_name
        self.max_context_length = max_context_length

    def _call_api(self, texts: List[str], text_type: str) -> List[List[float]]:
        response = requests.post(self.url, json={
            "texts": texts,
            "model_name": self.model_name,
            "max_context_length": self.max_context_length,
            "normalize_embeddings": True,
            "text_type": text_type,
            "manual_query_prefix": "search_query: ",
            "manual_passage_prefix": "search_document: ",
        }, timeout=120)
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            embeddings.extend(self._call_api(texts[i:i + batch_size], "passage"))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text], "query")[0]


def _build_embeddings(api_type: str) -> Embeddings:
    if api_type == "azure":
        custom_url = os.getenv("CUSTOM_EMBEDDING_URL")
        if custom_url:
            return CustomEmbeddings(
                url=custom_url,
                model_name=os.getenv("CUSTOM_EMBEDDING_MODEL", "bge-base-en-v1.5"),
            )
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            chunk_size=1000,
        )
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def _build_llm(api_type: str):
    if api_type == "azure":
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.3,
        )
    return ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


class DocumentSummarizer:
    def __init__(self):
        api_type = os.getenv("OPENAI_API_TYPE", "openai").lower()
        self.embeddings = _build_embeddings(api_type)
        self.llm = _build_llm(api_type)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def _extract_text(self, file_content: bytes, file_type: str) -> str:
        if file_type == "pdf":
            reader = PdfReader(io.BytesIO(file_content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            return file_content.decode("latin-1")

    def _create_vector_store(self, text: str) -> tuple[Chroma, int]:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR, exist_ok=True)

        chunks = self.text_splitter.split_text(text)
        store = Chroma.from_texts(chunks, self.embeddings, persist_directory=CHROMA_DIR)
        return store, len(chunks)

    def _build_chain(self, vector_store: Chroma, num_chunks: int):
        k = min(max(4, num_chunks // 3), 20)
        retriever = vector_store.as_retriever(search_kwargs={"k": k})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        callbacks = [langfuse_handler] if langfuse_handler else None
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | SUMMARY_PROMPT
            | self.llm.with_config(callbacks=callbacks)
            | StrOutputParser()
        )

    def summarize(self, file_content: bytes, file_type: str) -> str:
        text = self._extract_text(file_content, file_type)
        if not text.strip():
            raise ValueError("No text could be extracted from the document.")
        vector_store, num_chunks = self._create_vector_store(text)
        chain = self._build_chain(vector_store, num_chunks)
        return chain.invoke("Please provide a comprehensive summary of this document.")


def _check_config() -> bool:
    api_type = os.getenv("OPENAI_API_TYPE", "openai").lower()
    if api_type == "azure":
        required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                     "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME"]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            st.error(f"Azure OpenAI config incomplete. Missing: {', '.join(missing)}")
            return False
        st.success("Using Azure OpenAI")
    else:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key not found. Set OPENAI_API_KEY in your .env file.")
            return False
        st.success("Using OpenAI")
    return True


def _render_sidebar():
    with st.sidebar:
        st.header("About")

        api_type = os.getenv("OPENAI_API_TYPE", "openai").lower()
        if api_type == "azure":
            chat = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
            custom_url = os.getenv("CUSTOM_EMBEDDING_URL")
            embed = os.getenv("CUSTOM_EMBEDDING_MODEL", "bge-base-en-v1.5") if custom_url \
                else os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
            st.info(f"**Azure OpenAI**\n- Chat: `{chat}`\n- Embeddings: `{embed}`")
        else:
            chat = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
            embed = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
            st.info(f"**OpenAI**\n- Chat: `{chat}`\n- Embeddings: `{embed}`")

        st.markdown("""
        This AI Document Summarizer uses:
        - **RAG Framework** for context-aware summarization
        - **LangChain** for orchestrating AI workflows
        - **ChromaDB** for vector storage
        - **LangFuse** for observability (optional)
        """)

        st.header("Supported Formats")
        st.markdown("- PDF files (.pdf)\n- Text files (.txt)")


def main():
    st.set_page_config(page_title="AI Document Summarizer", page_icon="📄", layout="wide")
    st.title("AI Document Summarizer")
    st.markdown("Upload a document and get an AI-powered summary using RAG")

    if not _check_config():
        return

    if "summarizer" not in st.session_state:
        st.session_state.summarizer = DocumentSummarizer()

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"],
                                     help="Upload a PDF or text file to summarize")

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        file_type = uploaded_file.name.rsplit(".", 1)[-1].lower()

        if st.button("Generate Summary", type="primary"):
            with st.spinner("Processing document and generating summary..."):
                try:
                    summary = st.session_state.summarizer.summarize(
                        uploaded_file.read(), file_type
                    )
                    st.session_state.summary = summary
                    st.session_state.summary_filename = uploaded_file.name
                except Exception as e:
                    st.session_state.pop("summary", None)
                    st.error(f"An error occurred: {e}")

        if "summary" in st.session_state:
            st.subheader("Document Summary")
            st.markdown(st.session_state.summary)
            st.download_button(
                label="Download Summary",
                data=st.session_state.summary,
                file_name=f"summary_{st.session_state.summary_filename}.txt",
                mime="text/plain",
            )

    _render_sidebar()


if __name__ == "__main__":
    main()
