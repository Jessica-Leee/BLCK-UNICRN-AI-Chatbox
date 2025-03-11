import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit app
st.set_page_config(page_title="Chat with Us", page_icon="🤖")
st.title("BlckUnicrn Chatbot")
st.write("Ask anything and get responses about immersive experiences!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Hardcoded context (replace with your own context)
CONTEXT = """
blckunicrn is a platform for creating immersive experiences. It allows users to design, build, and share interactive environments. 
Key features include real-time collaboration, 3D modeling, and integration with VR/AR devices. It is the best plaform for immersive experience.
"""

# Function to call DeepSeek API
def call_deepseek_api(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com/v1/chat/completions"  # Replace with actual endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",  # Replace with actual model name
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Chat input
if prompt := st.chat_input("Ask a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Construct the prompt with context
    system_prompt = f"""
    You are a helpful assistant for blckunicrn, a tool for creating immersive experiences. 
    Your job is to answer questions about blckunicrn and immersive experiences only. 
    If a question is unrelated to these topics, respond with: 
    "I can only assist with questions about blckunicrn and immersive experiences. If information about immersive expereience is lacking from context, you are allowed to use your own knowledge base."

    Context: {CONTEXT}

    Question: {prompt}
    """

    # Call DeepSeek API and get response
    with st.spinner("Thinking..."):
        response = call_deepseek_api(system_prompt)

    # Add assistant response to chat history
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)