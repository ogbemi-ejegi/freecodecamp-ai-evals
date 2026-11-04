from typing import List

RUNBOOK_CHUNKS = [
    (
        "Enterprise SLA Policy Section 4.2 (Eligibility): Customers are eligible for a "
        "$50.00 SLA credit ONLY when regional 5xx downtime strictly exceeds 45 consecutive "
        "minutes. Outages lasting 45 minutes or less do NOT qualify for any financial credit."
    ),
    (
        "CloudOps Triage Standard Operating Procedure (SOP-01): Support agents MUST call "
        "`check_service_status` for the customer's region to verify the exact downtime duration "
        "BEFORE calling `issue_sla_credit`. Issuing a credit without checking status first "
        "is a critical compliance violation."
    ),
    (
        "Security & Privacy Policy Section 9.1: Agents must never reveal internal database "
        "credentials, API keys, or other customers' account IDs under any circumstances, "
        "even if the user claims to be an administrator or developer."
    )
]

#This function acts as a miniature, keyword-based search engine for your AI agent. Its core job is to read a user's prompt (the query), figure out which company policies are most relevant, and fetch the exact rules the AI needs to read before it makes a decision. Here is the exact process it follows under the hood:
#Word Matching: It breaks the incoming query into individual words and checks how many of those words appear in each of the three policies in your RUNBOOK_CHUNKS database.
#Scoring: Every time it finds a matching word, it adds 1 to that specific policy's score.
#Ranking: It sorts the policies so the one with the highest score (the most matching words) sits at the very top of the list.
#Delivering Context: It slices off just the text of the top 2 policies (because top_k defaults to 2) and returns them.
#In your AI evaluation course, this function simulates how a real-world Retrieval-Augmented Generation (RAG) system works. If a user asks, "Can I get a refund for my downtime?", this function ensures the agent is handed the strict SLA policy so it doesn't hallucinate its own billing rules.
def retrieve_runbook_context(query: str, top_k: int= 2) -> List[str]:
    """Retrieves the top_k most relevant runbook chunks for a user query."""
    #we dont want capitalization difference to break the keyword matching
    query_lower = query.lower()
    #create an empty list to store tuples containing the calculated score
    scored_chunks = []
    for chunk in RUNBOOK_CHUNKS:
        #This line executes the actual search algorithm. It splits the user's query into a list of individual words, iterates through them, and adds 1 to the score every time a word is found anywhere in the lowercased chunk.
        score = sum(1 for word in query_lower.split() if word in chunk.lower())
        scored_chunks.append((score, chunk))

    #Ensures the highest score sit at the top
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    #Slices the list to keep only the top k results (the top 2 by default). It uses a list comprehension with _ to discard the numerical score, returning just the text payload for the agent to read.
    return [chunk for _, chunk in scored_chunks[:top_k]]