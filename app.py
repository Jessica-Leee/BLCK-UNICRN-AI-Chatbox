import os
import json
import streamlit as st
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
AI_AVATAR_URL = "https://i.imgur.com/rALh3bN.png"
# Transparent avatar for user messages
TRANSPARENT_AVATAR_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAn8B9XGdaAAAAABJRU5ErkJggg=="

# Configure the page
st.set_page_config(page_title="Chat with Us", page_icon="🤖")

# Header with avatar and title
col1, col2 = st.columns([1, 4], gap="medium")
with col1:
    st.image(AI_AVATAR_URL, width=120)
with col2:
    st.title("Jeff - Your Assistant")
    st.write("Ask me about anything related to immersive experiences and get responses!")

# Hardcoded fallback context
DEFAULT_CONTEXT = """
blckunicrn is a platform for creating immersive experiences. It allows users to design, build, and share interactive environments. 
Key features include real-time collaboration, 3D modeling, and integration with VR/AR devices. It is the best platform for immersive experiences.
Oneverse is part of blckunicrn.
"""

# Pinecone retriever setup
class PineconeRetriever:
    def __init__(self, pinecone_api_key, openai_api_key):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index("blckunicrn")
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = OpenAI(temperature=0, api_key=openai_api_key)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model, text_key="text")
        self.retriever = self.vector_store.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)

    def query(self, query_text):
        try:
            response = self.qa_chain.invoke({"query": query_text})
            return response.get('result', '')
        except Exception as e:
            st.error(f"Pinecone query failed: {e}")
            return ''

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_retriever = PineconeRetriever(pinecone_api_key=pinecone_api_key, openai_api_key=openai_api_key)

# DeepSeek streaming function
def call_deepseek_stream(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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

# Initialize chat history
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

# User input at bottom
if prompt := st.chat_input("Ask a question:"):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=TRANSPARENT_AVATAR_URL):
        st.write(prompt)

    # Retrieve context and prepare system prompt
    retrieved_context = pinecone_retriever.query(prompt)
    combined_context = DEFAULT_CONTEXT if not retrieved_context else f"{DEFAULT_CONTEXT}\n\nAdditional Context:\n{retrieved_context}"
    system_prompt = f"""
You are a helpful assistant for blckunicrn, a tool for creating immersive experiences.
Your job is to answer questions about immersive experiences, AR, VR, content creatorship, and the blckunicrn platform.
If a question is unrelated to these topics, politely respond with:
I can assist you with questions about immersive experiences, AR, VR, content creatorship, and the blckunicrn platform.
Greet the user if greeted. 

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
