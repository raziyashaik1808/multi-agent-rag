import os
import time
import json
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types


# ============================================================
# 1. SETUP CHROMADB + LOCAL EMBEDDINGS
# ============================================================

client = chromadb.Client()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="company_knowledge_base",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# 2. SAMPLE KNOWLEDGE BASE
# ============================================================

documents = [
    "Project Apollo was initiated in Q1 2024 to modernize the analytics stack using Apache Spark.",
    "Internal code deployments are handled by SkyShip every Tuesday and Thursday at 10:00 AM UTC.",
    "Employees are eligible for a $500 annual home-office stipend after completing their 90-day probationary period.",
    "Travel expense reimbursement reports must be submitted within 14 business days of trip completion."
]

collection.add(
    documents=documents,
    metadatas=[
        {"dept": "Eng"},
        {"dept": "DevOps"},
        {"dept": "HR"},
        {"dept": "Finance"}
    ],
    ids=["doc_1", "doc_2", "doc_3", "doc_4"]
)

print(f"✓ Ingested {collection.count()} documents into ChromaDB.\n")


# ============================================================
# 3. GEMINI CLIENT
# ============================================================

ai_client = genai.Client()


# ============================================================
# 4. GEMINI HELPER
# ============================================================

def call_gemini(prompt: str) -> str:
    """
    Calls Gemini and retries temporary 503 errors.
    Returns a fallback message instead of crashing the whole pipeline.
    """

    for attempt in range(3):
        try:
            response = ai_client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )

            return response.text.strip()

        except Exception as e:
            if "503" in str(e):
                if attempt < 2:
                    print(
                        f"⚠ Gemini temporarily unavailable. "
                        f"Retrying ({attempt + 1}/3)..."
                    )
                    time.sleep(5)
                else:
                    print("⚠ Gemini unavailable after 3 attempts.")
                    return "Gemini is temporarily unavailable. Please try again later."

            else:
                raise


# ============================================================
# 5. RESEARCHER AGENT
# ============================================================

def researcher_agent(query: str) -> list[str]:
    """
    Researcher Agent:
    Analyzes the user query and creates smaller sub-queries
    that can be searched against the vector database.
    """

    prompt = f"""
You are the Researcher Agent in a Multi-Agent RAG system.

Your job is to analyze the user's question and break it into
the smallest useful research sub-queries.

User Question:
{query}

Create 1 to 3 concise sub-queries that should be searched
in a company knowledge base.

Return ONLY valid JSON in this format:

{{
    "sub_queries": [
        "sub-query 1",
        "sub-query 2"
    ]
}}

Do not answer the user's question.
Only create the research sub-queries.
"""

    response = call_gemini(prompt)

    try:
        data = json.loads(response)
        sub_queries = data.get("sub_queries", [])

        if not sub_queries:
            return [query]

        return sub_queries

    except json.JSONDecodeError:
        # Fallback if Gemini doesn't return perfect JSON
        return [query]


# ============================================================
# 6. CHROMADB TOOL
# ============================================================

def chromadb_search_tool(query: str, top_k: int = 2) -> list[str]:
    """
    Tool used by the agents to search the ChromaDB vector database.
    """

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    return results["documents"][0]


# ============================================================
# 7. WRITER AGENT
# ============================================================

def writer_agent(
    original_query: str,
    research_results: list[dict]
) -> str:
    """
    Writer Agent:
    Consolidates all information retrieved by the ChromaDB tool
    and produces the final answer.
    """

    research_text = ""

    for item in research_results:
        research_text += f"\nResearch Query: {item['query']}\n"

        for result in item["results"]:
            research_text += f"- {result}\n"

    prompt = f"""
You are the Writer Agent in a Multi-Agent RAG system.

Your task is to answer the user's original question using
ONLY the information retrieved from the knowledge base.

Original Question:
{original_query}

Retrieved Research:
{research_text}

Rules:
1. Use only the retrieved information.
2. Do not invent facts.
3. If the retrieved information is insufficient, say:
   "I don't have enough information in the provided knowledge base."
4. Give a concise and direct answer.

Final Answer:
"""

    return call_gemini(prompt)


# ============================================================
# 8. MULTI-AGENT RAG PIPELINE
# ============================================================

def ask_multi_agent_rag(question: str):

    print("=" * 70)
    print(f"Query: {question}")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: RESEARCHER AGENT
    # --------------------------------------------------------

    print("\n[1] Researcher Agent")

    sub_queries = researcher_agent(question)

    print("Generated sub-queries:")

    for i, sub_query in enumerate(sub_queries, 1):
        print(f"  [{i}] {sub_query}")


    # --------------------------------------------------------
    # STEP 2: CHROMADB TOOL
    # --------------------------------------------------------

    print("\n[2] ChromaDB Tool")

    research_results = []

    for sub_query in sub_queries:

        results = chromadb_search_tool(
            sub_query,
            top_k=2
        )

        research_results.append({
            "query": sub_query,
            "results": results
        })

        print(f"\nSearch: {sub_query}")

        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result}")


    # --------------------------------------------------------
    # STEP 3: WRITER AGENT
    # --------------------------------------------------------

    print("\n[3] Writer Agent")

    final_answer = writer_agent(
        question,
        research_results
    )

    print("\nFinal Answer:")
    print(final_answer)

    print("\n" + "=" * 70)


# ============================================================
# 9. TESTS
# ============================================================

if __name__ == "__main__":

    # Test 1
    ask_multi_agent_rag(
        "When can developers deploy code and which tool is used?"
    )

    # Test 2
    ask_multi_agent_rag(
        "How much is the home office budget and when do I qualify?"
    )

    # Test 3
    ask_multi_agent_rag(
        "What is the capital of Australia?"
    )