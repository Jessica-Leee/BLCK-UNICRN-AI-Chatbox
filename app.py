import os
import streamlit as st
import requests
import os  # Already imported

# Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA

# Load environment variables
openai_api_key = st.secrets["OPENAI_API_KEY"]   # make sure this is set

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
pinecone_api_key = os.getenv("PINECONE_API_KEY")
print(pinecone_api_key)
pinecone_retriever = PineconeRetriever(pinecone_api_key=pinecone_api_key, openai_api_key=openai_api_key)

# Function to call DeepSeek API
def call_deepseek_api(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
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