from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from uuid import uuid4
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

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="nexus_docs",
    embedding=embeddings,
)

def text_splitting():
    reader = PdfReader("./files/test.pdf")
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
    #texts=text_splitting()

    #docs = [Document(page_content=t) for t in texts]
    #ids = [str(uuid4()) for _ in docs]

    #vectorstore.add_documents(documents=docs, ids=ids)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    docs = retriever.invoke("operations by who?")

    print(docs[0].page_content)

if __name__ == "__main__":
    main()