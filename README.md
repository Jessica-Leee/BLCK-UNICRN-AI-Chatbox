
# 📄 Project Documentation: Chatbot for Immersive Experiences

This project contains two main Python files:

- `data_processing.py` – Handles PDF processing, chunking, and vector embedding.
- `app.py` – A Streamlit web interface for interacting with a chatbot based on the processed data.

---

## 📁 File Overview

### `data_processing.py`
- Loads and processes multiple PDFs listed in the `SOURCES` list.
- Extracts text from PDFs using `PyMuPDF (fitz)`.
- Chunks text by token count using `tiktoken`.
- Embeds text into a Pinecone vector store using OpenAI embeddings.

### `app.py`
- A Streamlit UI that allows users to interact with the embedded data.
- Uses LangChain’s `RetrievalQA` for answering questions based on the Pinecone vector store.
- Displays chat history and responses dynamically.

---

## ⚙️ Environment Setup

This project relies on environment variables for API keys and configuration. These should be stored in a `.env` file located in the project root.

**`.env` Example:**
```env
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENV=your_pinecone_env
PINECONE_INDEX_NAME=your_index_name
```

> ⚠️ **Important:**  
> Make sure `.env` is added to your `.gitignore` file to avoid accidentally leaking your keys.

```bash
# .gitignore
.env
```

---

## 🔑 Where to Change API Keys

API keys are loaded using `os.getenv(...)`:

- In `data_processing.py`:
  ```python
  openai_api_key = os.getenv("OPENAI_API_KEY")
  ```

- In `app.py`:
  ```python
  openai_api_key = os.getenv("OPENAI_API_KEY")
  ```

If you want to switch environments or services, simply change the keys in your `.env` file — **no code modification is needed**.

---

## 🚀 Running the App

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare .env**
   - Create a `.env` file and insert your API keys as described above.

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

---

## 📝 Notes

- The `SOURCES` and `IDs` arrays in `data_processing.py` determine which PDFs are loaded. Add your PDFs there if needed.
- Make sure the names in `SOURCES` match actual filenames in your project directory.
- Ensure `pinecone` and `openai` services are correctly set up before running the app.
