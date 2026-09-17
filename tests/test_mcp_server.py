"""
Tests for the MCP server

Covers the agent workflow end to end: ask the server for SQL, run it on a
real SQL engine (DuckDB standing in for the warehouse), hand the aggregate
rows back, and check the verdict against the row-level analysis. Also covers
the stdio transport with a real subprocess and bearer-token auth over HTTP.
"""

import json
import os
import socket
import sys
import threading
import time

import pandas as pd
import pytest

pytest.importorskip("mcp")
duckdb = pytest.importorskip("duckdb")

from mcp import Client, StdioServerParameters

import mcp_server
from modules.ab_testing import ABTestingEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "multivariant_checkout_test.csv")
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def warehouse():
    con = duckdb.connect()
    con.execute(f"CREATE TABLE checkout AS SELECT * FROM read_csv_auto('{CSV}')")
    yield con
    con.close()


def payload(result):
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


async def call(client, tool, **arguments):
    return await client.call_tool(tool, arguments)


def run_sql(con, sql):
    return json.loads(con.execute(sql).df().to_json(orient="records"))


# =============== THE AGENT WORKFLOW ===============

@pytest.mark.integration
async def test_tools_are_described_for_an_agent():
    async with Client(mcp_server.server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "plan_experiment_query", "analyze_experiment", "check_sample_ratio",
        "plan_sample_size", "analyze_rows", "decide_with_guardrails",
    }
    for tool in tools.values():
        assert len(tool.description) > 80          # an agent only has this text to go on
    assert "control_label" in tools["analyze_experiment"].input_schema["required"]


@pytest.mark.integration
async def test_plan_query_run_in_warehouse_then_analyze(warehouse):
    async with Client(mcp_server.server) as client:
        plan = payload(await call(
            client, "plan_experiment_query", dialect="duckdb", table="checkout",
            assignment_col="variant", metric_col="revenue", unit_col="user_id"
        ))
        rows = run_sql(warehouse, plan["sql"])
        assert len(rows) == 3 and set(rows[0]) == {"arm", "n", "mean", "var"}      # aggregates only
        
        analysis = payload(await call(
            client, "analyze_experiment", rows=rows, control_label="control", mde_pct=2.0
        ))
    
    df = pd.read_csv(CSV)
    engine = ABTestingEngine()
    by_variant = {c["variant"]: c for c in analysis["comparisons"]}
    for variant, comparison in by_variant.items():
        expected = engine.t_test(df[df.variant == "control"].revenue.values, df[df.variant == variant].revenue.values)
        assert comparison["result"]["mean_difference"] == pytest.approx(expected["mean_difference"])
        assert comparison["result"]["p_value_raw"] == pytest.approx(expected["p_value"])
        assert comparison["result"]["p_value"] >= comparison["result"]["p_value_raw"]       # Holm applied
    
    assert analysis["correction"] == "holm"
    assert analysis["sample_ratio_check"]["has_srm"] is False
    assert by_variant["express_pay"]["decision"] == "SHIP"
    assert by_variant["one_page_checkout"]["decision"] == "KEEP RUNNING"
    assert "Business Interpretation" in by_variant["express_pay"]["interpretation"]


@pytest.mark.integration
async def test_cuped_through_the_warehouse_turns_a_null_into_a_win(warehouse):
    async with Client(mcp_server.server) as client:
        async def analyze(kind, **extra):
            plan = payload(await call(client, "plan_experiment_query", dialect="duckdb", table="checkout",
                                      assignment_col="variant", metric_col="revenue", analysis=kind, **extra))
            rows = [r for r in run_sql(warehouse, plan["sql"]) if r["arm"] != "express_pay"]
            out = payload(await call(client, "analyze_experiment", rows=rows, control_label="control", analysis=kind))
            return out["comparisons"][0]
        
        plain = await analyze("mean")
        cuped = await analyze("cuped", second_col="pre_revenue")
    
    assert plain["decision"] == "KEEP RUNNING" and not plain["result"]["significant"]
    assert cuped["decision"] == "SHIP" and cuped["result"]["variance_reduction_pct"] > 80


@pytest.mark.integration
async def test_guardrail_blocks_a_winning_variant(warehouse):
    async with Client(mcp_server.server) as client:
        async def metric_result(metric):
            plan = payload(await call(client, "plan_experiment_query", dialect="duckdb", table="checkout",
                                      assignment_col="variant", metric_col=metric))
            rows = [r for r in run_sql(warehouse, plan["sql"]) if r["arm"] != "one_page_checkout"]
            out = payload(await call(client, "analyze_experiment", rows=rows, control_label="control"))
            return out["comparisons"][0]["result"]
        
        decision = payload(await call(
            client, "decide_with_guardrails",
            primary=await metric_result("revenue"),
            guardrails={"page_load_ms": await metric_result("page_load_ms")},
            guardrail_higher_is_better={"page_load_ms": False},
        ))
    assert decision["decision"] == "DO NOT SHIP"
    assert decision["harmed_guardrails"] == ["page_load_ms"]


@pytest.mark.integration
async def test_proportion_analysis_and_broken_split():
    rows = [{"arm": "control", "n": 10000, "successes": 500}, {"arm": "treatment", "n": 8000, "successes": 480}]
    async with Client(mcp_server.server) as client:
        out = payload(await call(client, "analyze_experiment", rows=rows, control_label="control",
                                 analysis="proportion"))
        planned = payload(await call(client, "analyze_experiment", rows=rows, control_label="control",
                                     analysis="proportion", expected_split={"control": 5, "treatment": 4}))
    assert out["sample_ratio_check"]["has_srm"] is True
    assert out["comparisons"][0]["decision"] == "INVALID - FIX THE EXPERIMENT"
    assert planned["sample_ratio_check"]["has_srm"] is False
    assert planned["comparisons"][0]["decision"] == "SHIP"


@pytest.mark.integration
async def test_sample_size_and_raw_rows_tools():
    async with Client(mcp_server.server) as client:
        size = payload(await call(client, "plan_sample_size", mde_pct=10, baseline_rate=0.05))
        rows = payload(await call(client, "analyze_rows", control=[10, 12, 9, 11, 10, 13] * 20,
                                  treatment=[14, 15, 13, 16, 14, 17] * 20, method="mann_whitney"))
    assert size["n_control"] > 10000
    assert rows["decision"] == "SHIP" and "looks" not in rows["result"]


# =============== WHAT AN AGENT MUST NOT BE ABLE TO DO ===============

@pytest.mark.integration
@pytest.mark.parametrize("arguments,fragment", [
    (dict(dialect="postgres", table="checkout; DROP TABLE users", assignment_col="variant", metric_col="revenue"),
     "plain"),
    (dict(dialect="postgres", table="checkout", assignment_col="variant", metric_col="revenue) --"), "plain"),
    (dict(dialect="postgres", table="checkout", assignment_col="variant", metric_col="revenue", analysis="ratio"),
     "second_col"),
])
async def test_query_planner_refuses_injection_and_incomplete_requests(arguments, fragment):
    async with Client(mcp_server.server) as client:
        result = await client.call_tool("plan_experiment_query", arguments)
    assert result.is_error
    assert fragment in result.content[0].text


@pytest.mark.integration
async def test_bad_analysis_requests_are_tool_errors_not_crashes():
    rows = [{"arm": "control", "n": 100, "mean": 1.0, "var": 1.0}, {"arm": "treatment", "n": 100, "mean": 1.1, "var": 1.0}]
    async with Client(mcp_server.server) as client:
        unknown = await client.call_tool("analyze_experiment", {"rows": rows, "control_label": "baseline"})
        missing = await client.call_tool("analyze_experiment", {"rows": rows, "control_label": "control",
                                                                "analysis": "cuped"})
        too_big = await client.call_tool("analyze_rows", {"control": [1.0] * 50001, "treatment": [1.0, 2.0]})
    assert unknown.is_error and "baseline" in unknown.content[0].text
    assert missing.is_error and "mean_x" in missing.content[0].text
    assert too_big.is_error and "50,000" in too_big.content[0].text


# =============== TRANSPORTS ===============

@pytest.mark.integration
async def test_stdio_transport_with_a_real_subprocess():
    params = StdioServerParameters(command=sys.executable, args=[os.path.join(ROOT, "mcp_server.py")], cwd=ROOT)
    async with Client(params) as client:
        names = [t.name for t in (await client.list_tools()).tools]
        check = payload(await call(client, "check_sample_ratio", counts={"control": 1000, "treatment": 700}))
    assert "analyze_experiment" in names
    assert check["has_srm"] is True


@pytest.mark.integration
def test_http_requires_the_bearer_token():
    import httpx
    import uvicorn
    
    with pytest.raises(ValueError):
        mcp_server.http_app("")
    with pytest.raises(ValueError):
        mcp_server.http_app("short")
    
    token = "test-token-0123456789abcdef"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(mcp_server.http_app(token), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if srv.started:
                break
            time.sleep(0.05)
        url = f"http://127.0.0.1:{port}/mcp"
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        
        assert httpx.post(url, json=body, headers=headers).status_code == 401
        assert httpx.post(url, json=body, headers={**headers, "Authorization": "Bearer wrong"}).status_code == 401
        
        ok = httpx.post(url, json=body, headers={**headers, "Authorization": f"Bearer {token}"})
        assert ok.status_code == 200, ok.text
        assert "analyze_experiment" in ok.text
    finally:
        srv.should_exit = True
        thread.join(timeout=5)
