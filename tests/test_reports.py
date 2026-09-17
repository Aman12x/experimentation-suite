"""
Tests for report export
"""

import io

import numpy as np
import pandas as pd
import pytest

from modules.ab_testing import ABTestingEngine
from modules.causal_inference import CausalInferenceLab
from utils.decision import ship_decision
from utils.interpreters import StatisticalInterpreter
from utils.report_generator import ReportGenerator


@pytest.fixture
def ab_bundle():
    rng = np.random.default_rng(0)
    results = ABTestingEngine().t_test(rng.normal(100, 10, 500), rng.normal(104, 10, 500))
    return results, StatisticalInterpreter.interpret_ab_test_results(results), ship_decision(results)


@pytest.mark.unit
def test_markdown_report_leads_with_decision(ab_bundle):
    results, interpretation, decision = ab_bundle
    report = ReportGenerator().create_markdown_report(results, interpretation, 't-test', decision=decision)
    
    assert "Decision: SHIP" in report
    assert report.index("Decision: SHIP") < report.index("Business Interpretation")
    assert f"p_value: {results['p_value']}" in report
    assert "significant: True" in report


@pytest.mark.unit
def test_html_report_has_balanced_markup(ab_bundle):
    results, interpretation, decision = ab_bundle
    report = ReportGenerator().create_html_report(results, interpretation, 't-test', decision=decision)
    
    assert report.count("<strong>") == report.count("</strong>") > 0
    assert "**" not in report
    assert "##" not in report
    assert "<h3>" in report and "<li>" in report


@pytest.mark.unit
def test_html_report_escapes_user_supplied_text(ab_bundle):
    results, interpretation, _ = ab_bundle
    report = ReportGenerator().create_html_report(
        {**results, 'note': '<script>alert(1)</script>'}, interpretation, '<b>x</b>'
    )
    assert "<script>" not in report
    assert "<b>x</b>" not in report


@pytest.mark.unit
def test_excel_report_round_trips(ab_bundle):
    results, _, _ = ab_bundle
    data = pd.DataFrame({'group': ['a', 'b'], 'metric': [1.0, 2.0]})
    workbook = ReportGenerator().create_excel_report(results, data, 't-test')
    
    sheets = pd.read_excel(io.BytesIO(workbook.getvalue()), sheet_name=None)
    summary = sheets['Summary'].set_index('Metric')['Value']
    assert float(summary['p_value']) == pytest.approx(results['p_value'])
    assert list(sheets['Raw Data'].columns) == ['group', 'metric']


@pytest.mark.unit
def test_excel_report_includes_result_tables_but_not_matched_rows():
    rng = np.random.default_rng(1)
    x = rng.normal(size=600)
    df = pd.DataFrame({
        'treated': (rng.random(600) < 1 / (1 + np.exp(-x))).astype(int), 'x': x,
    })
    df['y'] = 2 * df.treated + x + rng.normal(size=600)
    results = CausalInferenceLab().propensity_score_matching(df, 'treated', 'y', ['x'])
    
    workbook = ReportGenerator().create_excel_report(results, df, 'PSM')
    sheets = pd.read_excel(io.BytesIO(workbook.getvalue()), sheet_name=None)
    assert 'balance_stats' in sheets
    assert 'matched_treated' not in sheets


@pytest.mark.unit
def test_timestamp_is_not_frozen_at_construction(monkeypatch):
    import utils.report_generator as module
    
    class Clock:
        ticks = iter(pd.to_datetime(['2026-01-01 10:00:00', '2026-01-01 10:00:05']))
        @classmethod
        def now(cls):
            return next(cls.ticks)
    
    monkeypatch.setattr(module, 'datetime', Clock)
    generator = ReportGenerator()
    assert generator.timestamp != generator.timestamp
