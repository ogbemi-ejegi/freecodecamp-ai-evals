import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from deepeval.metrics import (
    FaithfulnessMetric,
    ToolCorrectnessMetric,
    GEval,
)

from src.agent import run_cloudops_agent
from evals.openrouter_judge import OpenRouterJudge

judge = OpenRouterJudge()

# 1. RAG Grounding Metric (async_mode=False prevents parallel socket throttling on OpenRouter)
faithfulness_metric = FaithfulnessMetric(
    threshold=0.80,
    model=judge,
    include_reason=True,
    async_mode=False,
)

# 2. Agent Trajectory Metric
tool_trajectory_metric = ToolCorrectnessMetric(
    threshold=1.0,
    model=judge,
    should_consider_ordering=True,
    include_reason=True,
)

# 3. Domain Policy G-Eval Metric
sla_compliance_geval = GEval(
    name="SLA SOP-01 Compliance Gate",
    criteria=(
        "Verify that the agent checked the regional service status first, accurately stated "
        "the downtime in minutes, refused any SLA credit when downtime was 45 minutes or less, "
        "and only issued a $50 SLA credit when downtime strictly exceeded 45 minutes."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.80,
    model=judge,
    async_mode=False,
)

with open("datasets/golden_dataset.json", "r", encoding="utf-8") as f:
    GOLDEN_CASES = json.load(f)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
def test_cloudops_rag_and_trajectory(case):
    # FIXED MODE: buggy_mode=False verifies status first and obeys the >45m SLA rule!
    result = run_cloudops_agent(
        user_query=case["input"],
        region=case["region"],
        account_id=case["account_id"],
        buggy_mode=False,
    )

    actual_tools = [ToolCall(name=t.name) for t in result.tools_called]
    expected_tools = [ToolCall(name=t_name) for t_name in case["expected_tools"]]

    combined_grounding_context = result.retrieval_context + [
        f"Executed Tool {t.name} returned: {t.output}" for t in result.tools_called
    ]

    test_case = LLMTestCase(
        input=result.user_input,
        actual_output=result.actual_output,
        expected_output=case["expected_output"],
        retrieval_context=combined_grounding_context,
        tools_called=actual_tools,
        expected_tools=expected_tools,
    )

    # run_async=False runs metrics sequentially so OpenRouter never drops concurrent sockets
    assert_test(
        test_case,
        [tool_trajectory_metric, faithfulness_metric, sla_compliance_geval],
        run_async=False,
    )