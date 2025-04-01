import os
import tiktoken
import openai
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define PDF sources and their IDs
SOURCES = [
    "onverse.pdf", 
    "Book.pdf", 
    "sony_music.pdf", 
    "research_immersive_experience.pdf", 
    "audience_report_immersive.pdf",
    "Annual Report.pdf",
    "Article Paper.pdf",
    "Data vision.pdf",
    "Immersive Analytics.pdf",
    "immersive technology.pdf",
    "Outlook.pdf",
    "Slide Deck.pdf"
]

IDs = [
    "onverse", 
    "book", 
    "sony music", 
    "research", 
    "audience report",
    "Annual Report",
    "Article Paper",
    "Data vision",
    "Immersive Analytics",
    "immersive technology",
    "Outlook",
    "Slide Deck"
]

class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from {self.pdf_path}: {e}")
            return ""

class Generator:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("No OpenAI API key found in .env")
        
        # Configure OpenAI client
        openai.api_key = self.openai_api_key

    def chunk_text_by_tokens(self, text, chunk_size, encoding_name="cl100k_base"):
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(encoding.decode(chunk_tokens))
        
        return chunks

    def generate_embeddings(self, chunks):
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                response = openai.embeddings.create(
                    input=chunk,
                    model="text-embedding-ada-002"
                )
                embeddings.append(response.data[0].embedding)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(chunks)} chunks")
            except Exception as e:
                print(f"Error generating embedding for chunk {i}: {e}")
                # Add an empty embedding to maintain alignment with chunks
                embeddings.append([0] * 1536)
        
        return embeddings

    def process_text(self, text, chunk_size=800):
        print(f"Processing text with {len(text)} characters")
        chunks = self.chunk_text_by_tokens(text, chunk_size)
        print(f"Created {len(chunks)} chunks")
        embeddings = self.generate_embeddings(chunks)
        return chunks, embeddings

class PineconeStore:
    def __init__(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "blckunicrn"
        
        # Check and create index if it doesn't exist
        if not self.index_exists():
            print(f"Creating new index: {self.index_name}")
            self.create_index()
        else:
            print(f"Using existing index: {self.index_name}")

    def index_exists(self):
        try:
            return self.pc.list_indexes().get("indexes", []) and any(
                idx["name"] == self.index_name for idx in self.pc.list_indexes().get("indexes", [])
            )
        except Exception as e:
            print(f"Error checking if index exists: {e}")
            return False

    def create_index(self):
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            print(f"Successfully created index: {self.index_name}")
        except Exception as e:
            print(f"Error creating index: {e}")
            raise

    def save_vectors(self, vectors, metadata, chunks):
        if len(vectors) != len(chunks):
            print(f"Warning: Number of vectors ({len(vectors)}) doesn't match number of chunks ({len(chunks)})")
            
        index = self.pc.Index(self.index_name)
        batch_size = 100  # Process in batches to avoid API limits
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = []
            end_idx = min(i + batch_size, len(vectors))
            
            for j in range(i, end_idx):
                if j >= len(vectors) or j >= len(chunks):
                    continue
                    
                vector_id = f"{metadata['id']}_chunk_{j}"
                chunk_metadata = {
                    "id": vector_id,
                    "source": metadata["source"],
                    "chunk": j,
                    "text": chunks[j]
                }
                batch_vectors.append((vector_id, vectors[j], chunk_metadata))
            
            if batch_vectors:
                try:
                    index.upsert(vectors=batch_vectors)
                    print(f"Successfully uploaded batch {i//batch_size + 1} ({len(batch_vectors)} vectors)")
                except Exception as e:
                    print(f"Error uploading batch {i//batch_size + 1}: {e}")

def main():
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        print("Error: Missing required API keys. Please check your .env file.")
        return
    
    # Process each PDF
    for i in range(len(SOURCES)):
        try:
            source = SOURCES[i]
            doc_id = IDs[i]
            
            # Get the full path to the PDF
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pdf_path = os.path.join(script_dir, source)
            
            print(f"\nProcessing {source} (ID: {doc_id})...")
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                print(f"Error: File not found - {pdf_path}")
                continue
                
            # Load and extract text from PDF
            loader = PDFLoader(pdf_path)
            text = loader.extract_text()
            
            if not text:
                print(f"Warning: No text extracted from {source}")
                continue
                
            print(f"Text extracted: {len(text)} characters")
            
            # Generate embeddings
            generator = Generator()
            chunks, embeddings = generator.process_text(text, chunk_size=800)
            print(f"Generated {len(embeddings)} embeddings")
            
            # Store vectors in Pinecone
            vector_store = PineconeStore()
            vector_store.save_vectors(
                embeddings,
                {"id": doc_id, "source": source},
                chunks
            )
            print(f"Successfully processed {source}")
            
        except Exception as e:
            print(f"Error processing {SOURCES[i]}: {e}")

if __name__ == "__main__":
    main()