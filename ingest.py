from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_config = models.VectorParams(
    size=3072,
    distance=models.Distance.COSINE
)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

def text_splitting():
    reader = PdfReader("test.pdf")
    number_of_pages = len(reader.pages)
    page = reader.pages[0]
    text = page.extract_text

    extracted_page = page.extract_text(extraction_mode="layout", layout_mode_scale_weight=1.0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )

    texts = text_splitter.split_text(extracted_page)
    #print(len(texts))
    #print(texts[1])
    return texts
    
def qdrant_create_collection(name: str):
    qdrant_client.create_collection(
        collection_name=name,
        vectors_config=collection_config
    )
    return

def qdrant_get_collections():
    print(qdrant_client.get_collections())
    return

def main():
    '''
    text=text_splitting()

    vectorstore = InMemoryVectorStore.from_texts(
    text,
    embedding=embeddings,
    )

    retriever = vectorstore.as_retriever()
    retrieved_documents = retriever.invoke("operations by who?")
    print(retrieved_documents[0].page_content)

    '''
    qdrant_get_collections()

if __name__ == "__main__":
    main()