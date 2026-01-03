from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini"
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="nexus_docs",
    embedding=embeddings,
)

prompt = PromptTemplate.from_template(
    "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
)

def retrieve_top3(query: str):
    docs = vectorstore.similarity_search(query, k=3)
    return docs

def rag_chain(query: str):
    docs = retrieve_top3(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    formatted_prompt = prompt.format(context=context, question=query)
    
    response = model.invoke(formatted_prompt)
    
    return response.content

def main():
    try:
        while True:
            query = input("Enter your query: ")
            answer = rag_chain(query)
            print("\nAnswer:")
            print(answer)
    except KeyboardInterrupt:
        print("Exiting")

if __name__ == "__main__":
    main()