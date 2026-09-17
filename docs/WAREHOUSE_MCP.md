# Analyzing warehouse experiments with an AI agent (MCP)

`mcp_server.py` exposes the suite to AI agents over the Model Context Protocol. It is built to sit **next to** a warehouse MCP server, not to replace one: the warehouse server runs SQL, this server does statistics.

```
        "Did checkout_v2 win on revenue? Control is 'control', latency is a guardrail."
                                        │
                                      agent
                  ┌─────────────────────┴──────────────────────┐
     experimentation-suite (this repo)              warehouse MCP server
     plan_experiment_query ──── SQL ───────────────▶ execute SQL
     analyze_experiment   ◀──── 3 rows ───────────── one row per arm
                  │
     verdict: SHIP / DO NOT SHIP / KEEP RUNNING / STOP / INVALID, with reasons
```

## Why aggregates

A Welch t-test needs three numbers per arm: `n`, `mean`, `variance`. CUPED and ratio metrics need three more (`mean_x`, `var_x`, `cov`). A proportion test needs two counts. The warehouse computes those in one `GROUP BY` over any number of rows, and only those few numbers cross the wire.

- Scale: the statistics cost the same for ten thousand rows or ten billion.
- Privacy: no user-level row reaches the agent or this server.
- Exactness: the from-stats results equal the row-level results. `tests/test_from_stats_and_sql.py` runs the generated SQL on DuckDB and requires agreement to 1e-9 with the row-level analysis of the same file, including the collapse to one row per user and the removal of users seen in more than one arm.

- Checked on a real warehouse: on 17 Sep 2026 the BigQuery-dialect SQL for all four analyses (mean with unit collapse, CUPED, ratio, proportion) and the date-window filter was run in BigQuery against the sample table. Every result matched the row-level analysis of the same file, with a worst relative gap of 2e-12. BigQuery returns numbers as JSON strings; `rows_to_arms` converts them.

Bootstrap, Mann-Whitney and sequential tests need the values themselves. `analyze_rows` offers them for up to 50,000 values per arm.

## Tools

| Tool | Purpose |
|---|---|
| `plan_experiment_query` | The one aggregate SQL statement to run. Dialects: `bigquery`, `snowflake`, `postgres`, `duckdb`. Analyses: `mean`, `proportion`, `cuped`, `ratio`. |
| `analyze_experiment` | From the result rows: test per variant (Holm-corrected when there are several), sample-ratio check, ship decision with reasons, plain-English interpretation. |
| `decide_with_guardrails` | Combine a primary metric with guardrail metrics into one decision. |
| `check_sample_ratio` | Traffic split check from units per arm. |
| `plan_sample_size` | Units per arm for a target lift, from a baseline rate or mean and standard deviation. |
| `analyze_rows` | Bootstrap, Mann-Whitney, sequential or t-test on raw values, small samples only. |

`plan_experiment_query` accepts only plain identifiers and ISO dates. There is no free-text `WHERE`, so an agent cannot be talked into emitting arbitrary SQL through it. Bad input comes back as a tool error whose message says what to fix.

## Run it

### With Claude Code or Claude Desktop (stdio)

```bash
pip install -r requirements.lock
claude mcp add experimentation-suite -- python /absolute/path/to/experimentation-suite/mcp_server.py
```

or in a project's `.mcp.json`:

```json
{
  "mcpServers": {
    "experimentation-suite": {
      "command": "python",
      "args": ["/absolute/path/to/experimentation-suite/mcp_server.py"]
    }
  }
}
```

### Over HTTP

```bash
export MCP_AUTH_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python mcp_server.py --http --host 0.0.0.0 --port 8080      # endpoint: /mcp
```

Every request must send `Authorization: Bearer $MCP_AUTH_TOKEN`. The server refuses to start in HTTP mode without a token of at least 16 characters. It is stateless, so it can scale to zero.

## Pair it with a warehouse server

Any MCP server that can run a SQL statement works. One option is Google's open-source [MCP Toolbox for Databases](https://github.com/googleapis/genai-toolbox), which supports BigQuery, PostgreSQL, Snowflake, MySQL, ClickHouse, Trino and others and ships a prebuilt `execute_sql` tool. Follow its README to point it at your warehouse with a **read-only** credential, then register both servers with your agent.

## What a session looks like

> **You:** Analyze the `checkout_v2` experiment in `analytics.experiments.checkout`. Users were randomized 50/50, arm column is `variant`, control is `control`. Primary metric is `revenue`, summed per user. Use `pre_revenue` to reduce variance.

The agent then:

1. calls `plan_experiment_query(dialect="bigquery", table="analytics.experiments.checkout", assignment_col="variant", metric_col="revenue", unit_col="user_id", unit_agg="sum", analysis="cuped", second_col="pre_revenue")`
2. runs the returned SQL through the warehouse server and gets one row per arm
3. calls `analyze_experiment(rows=..., control_label="control", analysis="cuped", mde_pct=2)`
4. reports the decision, the reasons, and the interpretation

For a guardrail it repeats steps 1 to 3 with `metric_col="page_load_ms"` and passes both results to `decide_with_guardrails` with `guardrail_higher_is_better={"page_load_ms": false}`.

## Without an agent

The same two steps are plain REST endpoints: `POST /api/warehouse/query` returns the SQL, `POST /api/warehouse/analyze` takes the rows. And in Python:

```python
from modules.sql_templates import aggregate_sql, rows_to_arms
from utils.warehouse import analyze_aggregates

sql = aggregate_sql("postgres", "experiments.checkout", "variant", "revenue", unit_col="user_id")["sql"]
rows = [dict(r) for r in connection.execute(sql).mappings()]      # your own driver
report = analyze_aggregates(rows, control_label="control", mde_pct=2.0)
```
