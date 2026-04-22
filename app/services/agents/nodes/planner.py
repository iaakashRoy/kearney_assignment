"""
Planner node — decomposes the question and selects tools.
"""
from __future__ import annotations

import time

from app.core.logging import get_logger
from app.services import llm
from app.services.agents.helpers import short
from app.services.agents.prompts import PLANNER_PROMPT
from app.services.agents.state import AgentState, Plan
from app.services.agents.trace import emit
from app.utils.text import extract_json

logger = get_logger(__name__)


def planner_node(state: AgentState) -> dict:
    """Planner agent — decomposes the question and selects tools."""
    question = state["question"]
    qid = state.get("qid", "-")
    t0 = time.perf_counter()
    logger.info("[%s] ▶ PLANNER  question=%r", qid, short(question, 300))
    emit("node.start", node="planner", qid=qid, question=question)

    fallback_plan: Plan = {
        "rationale": "Planner unavailable — falling back to single-hop hybrid retrieval.",
        "sub_queries": [question],
        "use_retriever": True,
        "use_sql": False,
        "sql_question": None,
    }

    try:
        raw = llm.generate_text(PLANNER_PROMPT.format(question=question), max_tokens=600)
        data = extract_json(raw)
        if not data:
            raise ValueError("planner returned no JSON")
        sub_qs_raw = data.get("sub_queries") or []
        sub_queries = [s.strip() for s in sub_qs_raw if isinstance(s, str) and s.strip()]
        if not sub_queries:
            sub_queries = [question]
        # cap to 3 and ensure original question is represented
        sub_queries = list(dict.fromkeys(sub_queries))[:3]

        sql_q = data.get("sql_question")
        plan: Plan = {
            "rationale": str(data.get("rationale", "")),
            "sub_queries": sub_queries,
            "use_retriever": bool(data.get("use_retriever", True)),
            "use_sql": bool(data.get("use_sql", False)),
            "sql_question": sql_q if isinstance(sql_q, str) and sql_q.strip() else None,
        }
        if not plan["use_retriever"] and not plan["use_sql"]:
            # Defensive: if the LLM disabled everything, force RAG.
            plan["use_retriever"] = True
        if plan["use_sql"] and not plan["sql_question"]:
            plan["sql_question"] = question

        logger.info("[%s]   rationale     : %s", qid, short(plan["rationale"], 300))
        logger.info(
            "[%s]   tool selection: retriever=%s sql=%s%s",
            qid, plan["use_retriever"], plan["use_sql"],
            f" sql_question={short(plan['sql_question'], 200)!r}" if plan["use_sql"] else "",
        )
        for i, sq in enumerate(sub_queries, 1):
            logger.info("[%s]   sub_query[%d] : %s", qid, i, short(sq, 200))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] planner LLM failed (%s) — using fallback plan", qid, exc)
        emit("planner.fallback", qid=qid, error=str(exc))
        plan = fallback_plan
        logger.info("[%s]   rationale     : %s", qid, plan["rationale"])
        logger.info("[%s]   sub_query[1] : %s", qid, short(question, 200))

    emit(
        "planner.plan",
        qid=qid,
        rationale=plan["rationale"],
        sub_queries=plan["sub_queries"],
        use_retriever=plan["use_retriever"],
        use_sql=plan["use_sql"],
        sql_question=plan.get("sql_question"),
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[%s] ✓ PLANNER done in %.0fms", qid, elapsed)
    emit("node.done", node="planner", qid=qid, elapsed_ms=int(elapsed))
    return {
        "plan": plan,
        "pending_sub_queries": list(plan["sub_queries"]),
        "iteration": 0,
        "components": [],
        "retrievals": [],
        "sql_attempts": [],
        "errors": [],
    }
