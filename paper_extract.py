import os
from dotenv import load_dotenv
import requests
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import time

# Load environment variables
load_dotenv(dotenv_path='.env')

# Set API Keys
SERP_API_KEY = os.getenv("SERP_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize Mistral Embeddings
mistral_embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=MISTRAL_API_KEY)

def get_mistral_embedding(text):
    """Gets vector embeddings using LangChain's Mistral integration."""
    embeddings = mistral_embeddings.embed_query(text)
    print(embeddings)
    return embeddings


def get_serpapi_results(query):
    """Fetches research papers from Google Scholar via SerpApi."""
    url = "https://serpapi.com/search"
    params = {"engine": "google_scholar", "q": query, "api_key": SERP_API_KEY}

    if not SERP_API_KEY:
        print("Error: SERP_API_KEY is not set. Please check your .env file.")
        return []

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error fetching from SerpApi:", response.text)
        return []

    data = response.json()
    papers = []
    if "organic_results" in data:
        for result in data["organic_results"]:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            papers.append({"title": title, "summary": snippet, "link": link})

    return papers

def store_papers_in_faiss(papers):
    """Stores research papers in FAISS for fast similarity search."""
    texts = [paper["title"] + " " + paper.get("summary", "") for paper in papers]
    
    # Create Document objects with text and metadata
    documents = []
    for i, (text, paper) in enumerate(zip(texts, papers)):
        # Store the text and paper info in the Document
        doc = Document(
            page_content=text,
            metadata={
                "index": i,
                "title": paper["title"],
                "summary": paper.get("summary", ""),
                "link": paper.get("link", "")
            }
        )
        documents.append(doc)
    
    # Create FAISS index from documents
    vectorstore = FAISS.from_documents(documents, mistral_embeddings)
    
    return vectorstore

def get_top_papers(query, vectorstore, k=5):
    """Finds top k relevant papers using FAISS similarity search."""
    # Search by query text directly
    docs = vectorstore.similarity_search(query, k=k)
    
    # Extract papers directly from document metadata
    matched_papers = []
    for doc in docs:
        matched_papers.append({
            "title": doc.metadata["title"],
            "summary": doc.metadata["summary"],
            "link": doc.metadata["link"]
        })
    
    return matched_papers

def main():
    query = input("Enter your research query: ")

    print("\nFetching research papers...")
    papers = get_serpapi_results(query)

    if not papers:
        print("No papers found. Try a different query.")
        return

    # Store papers in FAISS
    vectorstore = store_papers_in_faiss(papers)

    # Get top results
    relevant_papers = get_top_papers(query, vectorstore)

    print("\nTop Relevant Papers:\n")
    output_lines = []
    for i, paper in enumerate(relevant_papers[:5]):
        output_lines.append(f"{i+1}. {paper['title']}\n  {paper['link']}\n")
        print(output_lines[-1])

    # Append output to a file instead of overwriting
    with open("relevant_papers.txt", "a", encoding="utf-8") as f:
        f.writelines(output_lines)

    print("Relevant papers appended to 'relevant_papers.txt'.")

if __name__ == "__main__":
    main()
