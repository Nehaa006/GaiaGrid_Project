import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_knowledge_base(directory_path="./knowledge"):
    """
    Loads text files from the knowledge directory and builds a local 
    vector database using HuggingFace embeddings (FREE).
    """
    documents = []
    
    # 1. Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"⚠️ Directory {directory_path} not found.")
        return None

    # 2. Load all .txt files from the folder
    print(f"📂 Loading documents from {directory_path}...")
    try:
        # We manually loop to ensure encoding is handled correctly
        for file in os.listdir(directory_path):
            if file.endswith('.txt'):
                full_path = os.path.join(directory_path, file)
                loader = TextLoader(full_path, encoding='utf-8')
                documents.extend(loader.load())
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return None

    if not documents:
        print("⚠️ No documents found in the knowledge folder.")
        return None

    # 3. Split the text into smaller chunks
    # This helps the AI find specific answers more accurately
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    print(f"✅ Success! Loaded {len(docs)} text chunks into Gaia's memory.")

    # 4. Initialize Local Embeddings (MiniLM is tiny, fast, and free)
    print("🧠 Initializing local embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Create and persist the Vector Database
    # This saves the "brain" into the gaia_memory folder
    try:
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./gaia_memory"
        )
        print("💾 Vector database built and saved to './gaia_memory'")
        return vector_db
    except Exception as e:
        print(f"❌ Error building vector DB: {e}")
        return None