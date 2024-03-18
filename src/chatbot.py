# -*- coding: utf-8 -*-
# created by makise, 2024/3/18

import streamlit as st


def main():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "ℹ️ You can get your API key from the OpenAI dashboard."

    st.title("💬 Chatbot")
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
    main()
