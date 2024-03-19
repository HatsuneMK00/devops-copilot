# -*- coding: utf-8 -*-
# created by makise, 2024/3/18

import streamlit as st
import os

from langchain_agent import my_agent


def generate_response(agent, query, chat_history):
    response = agent.invoke(
        {
            "input": query,
        },
        config={"configurable": {"session_id": "<foo>"}},
    )
    print(response)
    return response['output']


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
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = generate_response(my_agent, prompt, chat_history=st.session_state.messages)
        msg = response
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


if __name__ == '__main__':
    main()
