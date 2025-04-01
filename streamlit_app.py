import os
import streamlit as st
import requests

# Try to import dotenv, but handle the case where it's not installed
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if it exists
except ImportError:
    # If dotenv is not available, we'll use Streamlit secrets instead
    pass
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA

# Get API keys from environment variables or Streamlit secrets
def get_api_key(key_name):
    # Try to get from Streamlit secrets first
    if hasattr(st, "secrets") and key_name in st.secrets:
        return st.secrets[key_name]
    # Then try environment variables
    return os.getenv(key_name)

openai_api_key = get_api_key("OPENAI_API_KEY")
pinecone_api_key = get_api_key("PINECONE_API_KEY")
deepseek_api_key = get_api_key("DEEPSEEK_API_KEY")

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

class StreamlitApp:
    @staticmethod
    def setup():
        """Sets up the Streamlit app"""
        st.set_page_config(page_title="Chat with Us", page_icon="🤖")
        st.title("BlckUnicrn Chatbot")
        st.write("Ask anything and get responses about immersive experiences!")

# Initialize the Streamlit app
StreamlitApp.setup()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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
        pinecone.init(api_key=pinecone_api_key)
        self.index_name = "blckunicrn"
        self.index = pinecone.Index(self.index_name)

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

# Function to call DeepSeek API
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

# Chat input and processing
if prompt := st.chat_input("Ask a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Retrieve additional context from Pinecone
    retrieved_context = ""
    if pinecone_retriever:
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

    # Call DeepSeek API and get response
    with st.spinner("Thinking..."):
        response = call_deepseek_api(system_prompt)

    # Add assistant response to chat history
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)