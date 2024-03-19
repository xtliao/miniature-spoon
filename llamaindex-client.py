# pip install llama-index llama-index-llms-ollama llama-index-llms-openai llama-index-readers-file docx2txt
# pip install streamlit watchdog
# pip install llama-index-embeddings-huggingface transformers
# pip install llama-index-vector-stores-chroma chromadb
# pip install llama-index-vector-stores-astra-db astrapy

import os
import chromadb
import streamlit as st

from pathlib import Path
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode

from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.astra_db import AstraDBVectorStore

st.title = "LLamaIndex Resume Reader"

# left side sidebar
with st.sidebar:
    st.header("LLamaIndex Resume Reader", divider=True)
    
    model_selected = st.radio("Choose a model:", ["llama2", "mistral", "gpt-4"], captions=["local model", "open source model", "commercial model"])
    temperature_selected = st.slider("Choose a temperature value:", 0.0, 2.0, 0.1)
    llm = Ollama(model="llama2", temperature=temperature_selected)
    if model_selected == "mistral":
        llm = Ollama(model="mistral", temperature=temperature_selected)
    elif model_selected == "gpt-4":
        llm = OpenAI(model="gpt-4", temperature=temperature_selected)

    embed_selected = st.radio("Choose an embedding model:", ["HuggingFace", "OpenAI"], captions=["open source", "commercial"], horizontal=True)
    embed = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    if embed_selected.lower() == "openai":
        embed = OpenAIEmbedding()
    
    st.divider()
    store_selected = st.radio("Choose a vector store:", ["In Memory", "Local ChromaDB", "Cloud Astradb"], horizontal=True)
    store_selected = store_selected.replace(" ", "-").lower()
    st.divider()
    data_path = "./data"
    uploaded_file = st.file_uploader("Upload a file:",type=["pdf", "docx", "txt"])
    if uploaded_file:
        full_path = Path(data_path) / uploaded_file.name
        with open(full_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        documents = SimpleDirectoryReader(data_path).load_data()
        
        if store_selected == "in-memory":
            index = VectorStoreIndex.from_documents(documents, embed=embed)
          
        elif store_selected == "local-chromadb":
            # save to db
            db1 = chromadb.PersistentClient(path = "chromadb/")
            collection = db1.get_or_create_collection("resume")
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, embed_model=embed)

            # load from db
            db2=chromadb.PersistentClient(path="chromadb/")
            collection=db2.get_or_create_collection("resume")
            vector_store=ChromaVectorStore(chroma_collection=collection)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store,embed_model=embed)

        elif store_selected == "cloud-astradb":
            ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
            ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")   
            assert ASTRA_DB_APPLICATION_TOKEN is not None and ASTRA_DB_APPLICATION_TOKEN !=""
            assert ASTRA_DB_API_ENDPOINT is not None and ASTRA_DB_API_ENDPOINT !=""

            vector_store = AstraDBVectorStore(
                collection_name="resume",
                token=ASTRA_DB_APPLICATION_TOKEN,
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                embedding_dimension=768, # size of BAAI/bge-base-en-v1.5 is 768
                
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                embed_model=embed,
                store_nodes_override=True)
         
        if model_selected == "gpt-4":
            chat_engine = index.as_chat_engine(llm=llm, chat_mode=ChatMode.OPENAI)
        else:
            chat_engine = index.as_chat_engine(llm=llm, chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT)

# right side chat
if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("say something"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if model_selected == "gpt-4":
            response = chat_engine.chat(prompt, tool_choice="query_engine_tool")
        elif model_selected == "llama2":
            response = chat_engine.chat(prompt)
        elif model_selected == "mistral":
            response = chat_engine.chat(prompt)

    models_env = f"The embedding model is: {embed_selected} while the LLM model is: {model_selected}."
    st.markdown(models_env)
    st.session_state.messages.append({"role": "assistant", "content": models_env})
    st.markdown(response.response) 
    st.session_state.messages.append({"role": "assistant", "content": response.response})