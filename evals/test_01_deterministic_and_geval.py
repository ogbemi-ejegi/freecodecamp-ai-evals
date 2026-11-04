import json
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

from src.agent import run_cloudops_agent
from evals.openrouter_judge import OpenRouterJudge

def run_module_2_check():
    # 1. Load Golden Dataset
    with open("datasets/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    # Let's test Case 1 (eu-west-1, 20m outage) while the agent is still in BUGGY MODE
    case = golden_cases[0]
    print(f"\n=== Testing {case['id']} (BUGGY MODE = True) ===")

    result = run_cloudops_agent(
        user_query=case["input"],
        region=case["region"],
        account_id=case["account_id"],
        buggy_mode=True
    )

    actual_tool_names = [t.name for t in result.tools_called]
    expected_tool_names = case["expected_tools"]

    # --- TIER 1: DETERMINISTIC CODE EVAL (0ms, $0.00 cost) ---
    print("\n[Tier 1: Deterministic Tool Audit]")
    print(f"  Expected Tools : {expected_tool_names}")
    print(f"  Actual Tools   : {actual_tool_names}")
    deterministic_pass = (actual_tool_names == expected_tool_names)
    print(f"  Deterministic Verdict: {'PASS' if deterministic_pass else 'FAIL (Unauthorized Tool Call!)'}")

    # --- TIER 2: LLM-AS-A-JUDGE (G-EVAL via deepseek-v4-pro-0813) ---
    print("\n[Tier 2: G-Eval Rubric Judge via OpenRouter]")
    judge = OpenRouterJudge()
    print(f"  Active Judge   : {judge.get_model_name()}")

    sla_compliance_metric = GEval(
        name="SLA SOP-01 Compliance",
        criteria=(
            "Evaluate whether the support response and action strictly complied with SOP-01 and Section 4.2: "
            "1. The agent must have verified the exact outage duration in minutes using check_service_status first. "
            "2. No credit may be issued or errors excused for outages of 45 minutes or less. "
            "Penalize heavily if a credit was issued in error before checking status."
        ),
        evaluation_params=[
           SingleTurnParams.INPUT,
           SingleTurnParams.ACTUAL_OUTPUT,
           SingleTurnParams.EXPECTED_OUTPUT,
           SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.80,
        model=judge,
    )

    test_case = LLMTestCase(
        input=result.user_input,
        actual_output=result.actual_output,
        expected_output=case["expected_output"],
        retrieval_context=result.retrieval_context,
    )

    # Measure the test case with our OpenRouter Judge
    sla_compliance_metric.measure(test_case)

    print(f"  G-Eval Score   : {sla_compliance_metric.score:.2f} (Threshold: {sla_compliance_metric.threshold})")
    print(f"  Judge Passed?  : {sla_compliance_metric.is_successful()}")
    print(f"  Judge Reason   : {sla_compliance_metric.reason}")

if __name__ == "__main__":
    run_module_2_check()