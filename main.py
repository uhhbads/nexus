from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
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
    check_compatibility=False  # REQUIRED for Qdrant Cloud
)

def retrieve_top3(query: str):
    # Convert query to vector
    query_vector = embeddings.embed_query(query)

    # Search Qdrant
    hits = client.http.search(
        collection_name="nexus_docs",
        query_vector=query_vector,
        limit=3,
        with_payload=True
    )

    results = []
    for hit in hits:
        results.append({
            "score": hit.score,
            "text": hit.payload.get("text", ""),
            "metadata": hit.payload
        })

    return results

def main():
    query = input("Enter your query: ")
    matches = retrieve_top3(query)

    print("\nTop 3 Matches:\n")
    for i, match in enumerate(matches, 1):
        print(f"{i}. Score: {match['score']:.4f}")
        print(match["text"])
        print("-" * 60)

if __name__ == "__main__":
    main()