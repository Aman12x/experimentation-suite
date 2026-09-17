#!/usr/bin/env python3
"""
MCP server for the Experimentation & Causal Analysis Suite

Lets an AI agent analyze an experiment that lives in a data warehouse without
moving row-level data. The agent pairs this server with any warehouse MCP
server (BigQuery, Snowflake, Postgres, ...):

    1. plan_experiment_query   -> the one aggregate SQL statement to run
    2. (agent runs it through the warehouse's own MCP server)
    3. analyze_experiment      -> test, sample-ratio check, decision, plain-English reading

Run locally over stdio (Claude Code, Claude Desktop):

    python mcp_server.py

Run over HTTP (requires a bearer token):

    MCP_AUTH_TOKEN=... python mcp_server.py --http --host 0.0.0.0 --port 8080
"""

import argparse
import functools
import hmac
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from modules.ab_testing import ABTestingEngine
from modules.from_stats import srm_from_counts
from modules.sql_templates import aggregate_sql
from utils.decision import ship_decision
from utils.warehouse import analyze_aggregates

MAX_ROWS_PER_ARM = 50_000

INSTRUCTIONS = """\
Statistical analysis of A/B tests, designed to work next to a warehouse MCP server.

Preferred flow for data in a warehouse:
1. Call plan_experiment_query to get one aggregate SQL statement. It returns one row per arm.
2. Run that SQL with the warehouse's own tool. Do not rewrite it; the column names are the contract.
3. Pass the result rows to analyze_experiment. It returns the test, a sample-ratio check,
   a ship decision with reasons, and a plain-English interpretation.

Ask the user for anything you cannot infer: which arm is control, whether a higher value of the
metric is good, the planned traffic split, and the unit that was randomized (usually a user id).
If the table has several rows per user, pass unit_col so rows are collapsed to one per user.

Use analyze_rows only for methods that need raw values (bootstrap, Mann-Whitney, sequential),
and only for small samples. Report decisions as given; do not soften INVALID or DO NOT SHIP.
"""

server = MCPServer(
    "experimentation-suite",
    title="Experimentation Suite",
    description="A/B test analysis from warehouse aggregates: tests, health checks and ship decisions",
    instructions=INSTRUCTIONS,
)
engine = ABTestingEngine()

Dialect = Literal['bigquery', 'snowflake', 'postgres', 'duckdb']
Analysis = Literal['mean', 'proportion', 'cuped', 'ratio']
Correction = Literal['holm', 'bonferroni', 'fdr_bh', 'none']


def explained(fn):
    """Bad input is the agent's mistake to read and correct, so its message must reach the agent.
    The SDK only forwards the text of a ToolError; anything else becomes a generic failure."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError) as exc:
            raise ToolError(str(exc)) from exc
    return wrapper


def _drop_bulky(result: Dict) -> Dict:
    """Interim looks are useful in a chart, not in an agent's context window"""
    return {k: v for k, v in result.items() if k != 'looks'}


@server.tool(title="Plan the warehouse query")
@explained
def plan_experiment_query(
    dialect: Dialect,
    table: str,
    assignment_col: str,
    metric_col: str,
    analysis: Analysis = 'mean',
    unit_col: Optional[str] = None,
    second_col: Optional[str] = None,
    unit_agg: Literal['mean', 'sum', 'max'] = 'mean',
    time_col: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the single aggregate SQL statement whose result analyze_experiment needs.

    analysis: 'mean' for continuous metrics (revenue, time on site); 'proportion' for yes/no
    metrics (converted); 'cuped' to reduce variance with a pre-experiment covariate given in
    second_col; 'ratio' for a ratio of two columns such as revenue per session, with the
    denominator in second_col.

    unit_col: the id of whatever was randomized (user_id). Give it whenever a unit can have
    several rows; rows are collapsed to one per unit and units seen in more than one arm are
    dropped. unit_agg says how a unit's rows combine: 'mean', 'sum' (revenue per user over
    the test), or 'max' (did it ever happen).

    Only plain identifiers are accepted; there is no free-text WHERE. Use time_col with
    start_date / end_date (ISO dates, inclusive) to restrict the window.
    """
    built = aggregate_sql(
        dialect, table, assignment_col, metric_col, analysis=analysis, unit_col=unit_col,
        second_col=second_col, unit_agg=unit_agg, time_col=time_col,
        start_date=start_date, end_date=end_date
    )
    built['analysis'] = analysis
    built['next_step'] = (
        "Run this SQL with the warehouse tool, then call analyze_experiment with the result rows "
        f"and analysis='{analysis}'."
    )
    return built


@server.tool(title="Analyze an experiment from aggregate rows")
@explained
def analyze_experiment(
    rows: List[Dict[str, Any]],
    control_label: str,
    analysis: Analysis = 'mean',
    higher_is_better: bool = True,
    alpha: float = 0.05,
    correction: Correction = 'holm',
    mde_pct: Optional[float] = None,
    expected_split: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run the test, the sample-ratio check and the ship decision from per-arm aggregates.

    rows: the result of the plan_experiment_query SQL, one dict per arm, e.g.
    [{"arm": "control", "n": 2010, "mean": 51.2, "var": 410.5}, {"arm": "treatment", ...}].
    For 'proportion' the columns are arm, n, successes. For 'cuped' and 'ratio' the rows also
    carry mean_x, var_x and cov.

    control_label: which arm is the baseline. higher_is_better: false for churn, latency, errors.
    mde_pct: smallest relative lift worth shipping; lets the decision tell 'keep running' from
    'stop, no meaningful effect'. expected_split: planned traffic share per arm, e.g.
    {"control": 0.5, "treatment": 0.5}; defaults to equal.

    With more than two arms every variant is compared to control and p-values are corrected.
    """
    return analyze_aggregates(
        rows, control_label, analysis=analysis, higher_is_better=higher_is_better, alpha=alpha,
        correction=correction, mde_pct=mde_pct, expected_split=expected_split
    )


@server.tool(title="Check the traffic split")
@explained
def check_sample_ratio(
    counts: Dict[str, int],
    expected_split: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Sample ratio mismatch check from units per arm, e.g. {"control": 10050, "treatment": 9920}.

    A mismatch means assignment or logging is broken and no result from the experiment can be
    trusted until it is explained. Uses alpha = 0.001, the convention for this check.
    """
    return srm_from_counts(counts, expected_split)


@server.tool(title="Plan the sample size")
@explained
def plan_sample_size(
    mde_pct: float,
    baseline_mean: Optional[float] = None,
    baseline_std: Optional[float] = None,
    baseline_rate: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, Any]:
    """Units needed per arm to detect a relative lift of mde_pct percent.

    For a yes/no metric give baseline_rate (0.05 for 5% conversion). For a continuous metric
    give baseline_mean and baseline_std.
    """
    if baseline_rate is not None:
        if not 0 < baseline_rate < 1:
            raise ValueError("baseline_rate must be between 0 and 1")
        baseline_mean, baseline_std = baseline_rate, (baseline_rate * (1 - baseline_rate)) ** 0.5
    if baseline_mean is None or baseline_std is None:
        raise ValueError("Give baseline_rate, or both baseline_mean and baseline_std")
    return engine.calculate_sample_size(baseline_mean, mde_pct, baseline_std, alpha=alpha, power=power)


@server.tool(title="Analyze raw values (small samples only)")
@explained
def analyze_rows(
    control: List[float],
    treatment: List[float],
    method: Literal['bootstrap', 'mann_whitney', 'sequential', 't_test'] = 'bootstrap',
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Tests that need the values themselves: bootstrap and Mann-Whitney for heavy-tailed
    metrics, sequential for data in arrival order. One value per randomized unit.

    Limited to 50,000 values per arm. For anything larger use plan_experiment_query and
    analyze_experiment, which never move rows.
    """
    if max(len(control), len(treatment)) > MAX_ROWS_PER_ARM:
        raise ValueError(
            f"At most {MAX_ROWS_PER_ARM:,} values per arm. Use plan_experiment_query + analyze_experiment instead."
        )
    run = {
        'bootstrap': engine.bootstrap_test, 'mann_whitney': engine.mann_whitney_test,
        'sequential': engine.sequential_test, 't_test': engine.t_test,
    }[method]
    result = _drop_bulky(run(control, treatment, alpha=alpha))
    decision = ship_decision(result, higher_is_better=higher_is_better)
    return {'decision': decision['decision'], 'reasons': decision['reasons'], 'result': result}


@server.tool(title="Decide with guardrails")
@explained
def decide_with_guardrails(
    primary: Dict[str, Any],
    guardrails: Dict[str, Dict[str, Any]],
    higher_is_better: bool = True,
    guardrail_higher_is_better: Optional[Dict[str, bool]] = None,
    mde_pct: Optional[float] = None,
    sample_ratio_check: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine a primary metric with guardrail metrics into one ship decision.

    Run analyze_experiment once per metric, then pass each comparison's 'result' object here,
    unmodified: primary is the decision metric, guardrails maps a metric name to its result.
    Do not trim or rebuild the objects; the direction of each effect is read from them.
    guardrail_higher_is_better marks direction per guardrail (false for latency, errors,
    refunds). A guardrail that moved significantly the wrong way blocks shipping.
    """
    return ship_decision(
        primary, guardrails=guardrails, higher_is_better=higher_is_better,
        guardrail_higher_is_better=guardrail_higher_is_better or {}, mde_pct=mde_pct,
        health={'sample_ratio_mismatch': sample_ratio_check} if sample_ratio_check else None
    )


class BearerAuth:
    """ASGI middleware: every HTTP request must carry the shared bearer token"""

    def __init__(self, app, token: str):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            supplied = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(supplied, self.expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
                return
        await self.app(scope, receive, send)


def http_app(token: str, host: str = "127.0.0.1"):
    """The streamable-HTTP ASGI app behind bearer-token auth"""
    if not token or len(token) < 16:
        raise ValueError("MCP_AUTH_TOKEN must be set to a secret of at least 16 characters")
    return BearerAuth(server.streamable_http_app(host=host, stateless_http=True, json_response=True), token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentation Suite MCP server")
    parser.add_argument("--http", action="store_true", help="Serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()

    if not args.http:
        server.run("stdio")
        return

    import uvicorn
    uvicorn.run(http_app(os.environ.get("MCP_AUTH_TOKEN", ""), host=args.host), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
