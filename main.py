import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import pickle
import requests
import fitz  

st.set_page_config(page_title="Scheme Research Tool", layout="wide")
st.title(" Scheme Research Tool")

# Helper Functions
def load_url_content(url):
    documents = []
    if url.endswith(".pdf"):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            text = ""
            with fitz.open("temp.pdf") as pdf:
                for page in pdf:
                    text += page.get_text()
            documents.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            st.error(f"Failed to load PDF from {url}: {e}")
    else:
        from langchain.document_loaders import UnstructuredURLLoader
        loader = UnstructuredURLLoader(urls=[url])
        documents.extend(loader.load())
    return documents

# Sidebar Input
with st.sidebar:
    st.header(" Input Scheme Article URLs")
    url_input = st.text_area("Enter one or more URLs (each in a new line):")
    process_button = st.button("Process URLs")

# Process URLs 
if process_button:
    if not url_input.strip():
        st.error("Please enter at least one URL.")
    else:
        url_list = [u.strip() for u in url_input.strip().split("\n") if u.strip()]
        all_docs = []
        with st.spinner(" Loading and processing..."):
            try:
                for url in url_list:
                    all_docs.extend(load_url_content(url))

                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(all_docs)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(chunks, embeddings)

                with open("faiss_store_openai.pkl", "wb") as f:
                    pickle.dump(vectorstore, f)

                st.success(" URLs processed and FAISS index saved!")
            except Exception as e:
                st.error(f" Error: {str(e)}")

#Ask Questions
st.header("Ask a Question About the Scheme")
question = st.text_input("Type your question here:")
ask_button = st.button(" Get Answer")

if ask_button and question:
    try:
        with open("faiss_index.pkl", "rb") as f:
            vectorstore = pickle.load(f)

        # Load local model
        qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=256)
        local_llm = HuggingFacePipeline(pipeline=qa_pipeline)

        qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        with st.spinner("Thinking..."):
            result = qa_chain({"query": question})
            st.subheader("📌 Answer:")
            st.write(result["result"])

            st.subheader("🔗 Source URLs:")
            sources = {doc.metadata.get("source", "Unknown") for doc in result["source_documents"]}
            for src in sources:
                st.write(f"- {src}")

    except FileNotFoundError:
        st.error(" Please process URLs first.")
    except Exception as e:
        st.error(f" Error: {str(e)}")
