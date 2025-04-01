import os
import tiktoken
import openai
import fitz
import pinecone
import os  # Already imported

# Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

SOURCES = ["onverse.pdf", "Book.pdf", "sony_music.pdf", "research_immersive_experience.pdf", "audience_report_immersive.pdf","Annual Report.pdf","Article Paper.pdf","Data vision.pdf","Immersive Analytics.pdf","immersive technology.pdf","Outlook.pdf","Slide Deck.pdf"]
IDs = ["onverse", "book", "sony music", "research", "audience report","Annual Report","Article Paper","Data vision","Immersive Analytics","immersive technology","Outlook","Slide Deck"]

class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

class Generator:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("No API key found in .env")
        self.openai_api_key = openai_api_key

    def chunk_text_by_tokens(self, text, chunk_size, encoding_name="cl100k_base"):
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return [encoding.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

    def generate_embeddings(self, chunks):
        embeddings = []
        for chunk in chunks:
            response = openai.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            # Access the embedding via attribute access as per the official response object
            embeddings.append(response.data[0].embedding)
        return embeddings

    def process_text(self, text, chunk_size=1000):
        chunks = self.chunk_text_by_tokens(text, chunk_size)
        embeddings = self.generate_embeddings(chunks)
        return chunks, embeddings

class PineconeStore:
    def __init__(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "blckunicrn"
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

    def save_vectors(self, vectors, metadata, chunks):
        index = self.pc.Index(self.index_name)
        for i, vector in enumerate(vectors):
            vector_id = f"{metadata['id']}_chunk_{i}"
            chunk_metadata = {
                "id": vector_id,
                "source": metadata["source"],
                "chunk": i,
                "text": chunks[i]
            }
            index.upsert(vectors=[(vector_id, vector, chunk_metadata)])


# Process the PDF and store vectors in Pinecone
for i in range(len(SOURCES)):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, SOURCES[i])
    loader = PDFLoader(pdf_path)
    text = loader.extract_text()
    print("text_extracted")
    generator = Generator()
    chunks, embeddings = generator.process_text(text, chunk_size=800)
    print("embeddings_converted")
    vector_store = PineconeStore()
    vector_store.save_vectors(embeddings, {"id": IDs[i], "source": SOURCES[i]}, chunks)
    print("vectors saved")
