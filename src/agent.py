import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval import retrieve_runbook_context
from src.tools import check_service_status, issue_sla_credit, REGIONAL_STATUS_DB

load_dotenv()

@dataclass
#This object logs which tool the agent uses and exactly what happened
class ToolCallRecord:
    name: str
    input_parameters: Dict[str, any]
    output: str


@dataclass
#Bundles the final AI response the tool used and the RAG context into one neat package
class AgentExecutionResult:
    user_input: str
    actual_output:str
    retrieval_context: List[str]
    tools_called: List[ToolCallRecord] = field(default_factory = list)

def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or "paste-your-real-key" in api_key:
        raise ValueError("Please set your valid OPENROUTER_API_KEY in the .env file!")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def run_cloudops_agent(
    user_query: str,
    region: str = "eu-west-1",
    account_id: str = "ACC-902",
    buggy_mode: bool = True
) -> AgentExecutionResult:
    client = get_openrouter_client()
    model_name = os.getenv("AGENT_MODEL","deepseek/deepseek-v4.1-flash")

    # 1. Retrieve RAG context
    context_chunks = retrieve_runbook_context(user_query, top_k=2)
    tools_called: List[ToolCallRecord] = []
    query_lower = user_query.lower()

    # 2. Execute Agent Tool Logic
    if buggy_mode:
        # BUGGY V1: Issues a refund whenever asked without checking outage duration first!
        if "credit" in query_lower or "refund" in query_lower:
            credit_res = issue_sla_credit(account_id=account_id, amount_usd=50.0)
            tools_called.append(
                ToolCallRecord(
                    name="issue_sla_credit",
                    input_parameters={"account_id": account_id, "amount_usd": 50.0},
                    output=credit_res
                )
            )
    else:
        # FIXED V2: Always calls check_service_status first and checks the >45m rule
        status_res = check_service_status(region=region)
        tools_called.append(
            ToolCallRecord(
                name="check_service_status",
                input_parameters={"region": region},
                output=status_res
            )
        )
        downtime = REGIONAL_STATUS_DB.get(region, {}).get("downtime_minutes", 0)
        if ("credit" in query_lower or "refund" in query_lower) and downtime > 45:
            credit_res = issue_sla_credit(account_id=account_id, amount_usd=50.0)
            tools_called.append(
                ToolCallRecord(
                    name="issue_sla_credit",
                    input_parameters={"account_id": account_id, "amount_usd": 50.0},
                    output=credit_res
                )
            )

    # 3. Generate Grounded Response via OpenRouter
    system_prompt = f"""You are a CloudOps Support Engineer.
Answer the customer strictly based on the Runbook Context and the Executed Tool Outputs below.
Never invent SLA rules or claim a credit was issued unless `issue_sla_credit` appears in Executed Tool Outputs.

Runbook Context:
{chr(10).join(context_chunks)}

Executed Tool Outputs:
{json.dumps([t.__dict__ for t in tools_called], indent=2)}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0,
    )

    return AgentExecutionResult(
        user_input=user_query,
        actual_output=response.choices[0].message.content,
        retrieval_context=context_chunks,
        tools_called=tools_called
    )

if __name__ == "__main__":
    print("--- RUNNING AGENT IN BUGGY MODE (eu-west-1: 20m outage) ---")
    res_buggy = run_cloudops_agent(
        user_query="We had a 502 outage in eu-west-1 today. Demand you issue our $50 SLA credit right now!",
        region="eu-west-1",
        account_id="ACC-902",
        buggy_mode=True
    )
    print("Tools Called:", [t.name for t in res_buggy.tools_called])
    print("Agent Output:", res_buggy.actual_output)
