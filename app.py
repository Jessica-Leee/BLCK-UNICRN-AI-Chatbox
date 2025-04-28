import os
import json
import streamlit as st
import requests
# Try to import dotenv, but handle the case where it's not installed
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if it exists
except ImportError:
    # If dotenv is not available, we'll use Streamlit secrets instead
    pass
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA


# Get API keys from environment variables or Streamlit secrets
def get_api_key(key_name):
    return os.getenv(key_name)


openai_api_key = get_api_key("OPENAI_API_KEY")
pinecone_api_key = get_api_key("PINECONE_API_KEY")
deepseek_api_key = get_api_key("DEEPSEEK_API_KEY")

# Add avatar URLs from second file
AI_AVATAR_URL = "https://i.imgur.com/rALh3bN.png"
TRANSPARENT_AVATAR_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAn8B9XGdaAAAAABJRU5ErkJggg=="

# Check for required API keys
if not all([openai_api_key, pinecone_api_key, deepseek_api_key]):
    missing_keys = []
    if not openai_api_key:
        missing_keys.append("OPENAI_API_KEY")
    if not pinecone_api_key:
        missing_keys.append("PINECONE_API_KEY")
    if not deepseek_api_key:
        missing_keys.append("DEEPSEEK_API_KEY")

    st.error(f"Missing required API keys: {', '.join(missing_keys)}. Please check your .env file.")
    st.stop()

# Configure the page
st.set_page_config(page_title="Chat with Us", page_icon="🤖")

# Header with avatar and title
col1, col2 = st.columns([1, 4], gap="medium")
with col1:
    st.image(AI_AVATAR_URL, width=120)
with col2:
    st.title("Jeff - Your Assistant")
    st.write("Ask me about anything related to immersive experiences and get responses!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "assistant":
        avatar = AI_AVATAR_URL
    else:
        avatar = TRANSPARENT_AVATAR_URL
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

# Hardcoded fallback context (replace with your own context)
DEFAULT_CONTEXT = """
blckunicrn is a platform for creating immersive experiences. It allows users to design, build, and share interactive environments. 
Key features include real-time collaboration, 3D modeling, and integration with VR/AR devices. It is the best platform for immersive experiences.
Oneverse is part of blckunicrn. 
"""

class PineconeRetriever:
    def __init__(self, pinecone_api_key, openai_api_key):
        # Initialize Pinecone connection
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "blckunicrn"
        self.index = self.pc.Index(self.index_name)

        # Initialize OpenAI model and embeddings
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = OpenAI(temperature=0, api_key=openai_api_key)

        # Create the Pinecone vector store and retriever
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model, text_key="text")
        self.retriever = self.vector_store.as_retriever()

        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)

    def query(self, query_text):
        # Execute the QA chain with the input query
        try:
            response = self.qa_chain.invoke({"query": query_text})
            return response.get('result', '')
        except Exception as e:
            st.error(f"Pinecone query failed: {e}")
            return ''

# Initialize the PineconeRetriever instance
try:
    pinecone_retriever = PineconeRetriever(pinecone_api_key=pinecone_api_key, openai_api_key=openai_api_key)
except Exception as e:
    st.error(f"Failed to initialize Pinecone retriever: {e}")
    pinecone_retriever = None


    # Function to call DeepSeek API (non-streaming)
    def call_deepseek_api(prompt):
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-chat",
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


    # DeepSeek streaming function
    def call_deepseek_stream(prompt):
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {deepseek_api_key}", "Content-Type": "application/json"}
        payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "stream": True}
        try:
            response = requests.post(url, json=payload, headers=headers, stream=True)
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                delta = chunk.get("choices", [])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
        except Exception as e:
            st.error(f"An error occurred during streaming: {e}")

# Chat input and processing
if prompt := st.chat_input("Ask a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Retrieve additional context from Pinecone
    retrieved_context = pinecone_retriever.query(prompt)
    
    # If retrieved_context is non-empty, append it to the default context.
    # Otherwise, use the default context alone.
    if retrieved_context:
        combined_context = f"{DEFAULT_CONTEXT}\n\nAdditional Context:\n{retrieved_context}"
    else:
        combined_context = DEFAULT_CONTEXT

    # Build system prompt using the combined context
    system_prompt = f"""
You are a helpful assistant for blckunicrn, a tool for creating immersive experiences.
Your job is to answer questions about immersive experiences, AR, VR, content creatorship, and the blckunicrn platform.
If a question is unrelated to these topics, check if it is related to what you know about blckunicrn. If it is not, respond with: 
"I can only assist with questions about immersive experiences, AR, VR, content creatorship, and the blckunicrn platform."
Use the provided context to inform your answer. 

Context:
{combined_context}

Question: {prompt}
"""

    # Assistant response
    with st.chat_message("assistant", avatar=AI_AVATAR_URL):
        placeholder = st.empty()
        full_reply = ""
        for token in call_deepseek_stream(system_prompt):
            full_reply += token
            placeholder.write(full_reply)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_reply})
