import streamlit as st
import os

st.title("BlckUnicrn Chatbot - Basic Version")
st.write("This is a basic version of the chatbot to verify Streamlit Cloud deployment works.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Simple chat functionality
if prompt := st.chat_input("Ask a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate simple response
    response = "This is a basic version of the BlckUnicrn chatbot. We're currently resolving dependency issues to get the full version working."
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# System information for debugging
st.sidebar.title("System Info")
st.sidebar.write(f"Python version: {os.sys.version}")
st.sidebar.write(f"Streamlit version: {st.__version__}")