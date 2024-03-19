# -*- coding: utf-8 -*-
# created by makise, 2024/3/20

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
import os


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


def get_retriever_tool(vector_store, embedding_fn):
    if not os.path.exists(vector_store):
        embed_knowledge_pdf(
            "Kubernetes Cookbook_ Building Cloud Native Applications-O'Reilly Media (2023)(Z-Lib.io).pdf",
            embedding_fn,
            "faiss"
        )
    vector = FAISS.load_local(vector_store, embedding_fn, allow_dangerous_deserialization=True)
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "kubernetes_knowledge",
        "Search for information about Kubernetes. For any questions about Kubernetes, you must use this tool!",
    )
    return retriever_tool


def get_agent_with_history(llm):
    print("get agent called")
    tools = [get_retriever_tool("faiss", OpenAIEmbeddings())]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(
        tools=tools,
        llm=llm,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    message_history = ChatMessageHistory()
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history


my_agent = get_agent_with_history(ChatOpenAI())
