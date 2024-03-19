# -*- coding: utf-8 -*-
# created by makise, 2024/3/18

import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def embed_knowledge_pdf(doc, embedding_fn, vector_store):
    loader = PyPDFLoader("../knowledge/{}".format(doc))
    pages = loader.load_and_split()
    file_index = FAISS.from_documents(pages, embedding_fn)

    if os.path.exists(vector_store):
        local_index = FAISS.load_local(vector_store, embedding_fn)
        local_index.merge_from(file_index)
        print("Merged local index with new file {}".format(doc))
        local_index.save_local(vector_store)
        print("Updated index saved")
    else:
        file_index.save_local(vector_store)
        print("New index created and saved")


def get_retriever(vector_store, embedding_fn):
    vector = FAISS.load_local(vector_store, embedding_fn, allow_dangerous_deserialization=True)
    retriever = vector.as_retriever()
    return retriever


def generate_response(query, llm, retriever):
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})
    print(response["answer"])
    return response["answer"]


def main():
    with st.sidebar:
        st.selectbox("Select Function", ["Chatbot", "More on the way"], index=0)

    st.title("💬 DevOps Copilot")
    st.caption("🚀 A DevOps copilot backed by OpenAI's GPT-3.5 and RAG.")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = "Hello World"
        msg = response
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


if __name__ == '__main__':
    generate_response(
        "What is book title of the kubernetes knowledge you are referring to?",
        ChatOpenAI(),
        get_retriever("faiss", OpenAIEmbeddings()))
