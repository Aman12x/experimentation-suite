"""
Ship Decision
Combines the primary metric, guardrail metrics, and health checks into one call
"""

from typing import Dict, List, Optional

from modules.ab_advanced import format_change


SHIP = "SHIP"
DO_NOT_SHIP = "DO NOT SHIP"
KEEP_RUNNING = "KEEP RUNNING"
STOP_NO_EFFECT = "STOP - NO MEANINGFUL EFFECT"
INVALID = "INVALID - FIX THE EXPERIMENT"


def _checked(result: Dict, name: str) -> Dict:
    """
    A result the decision can rely on: significance, p-value and the direction of the effect.
    
    Callers (agents especially) sometimes pass a trimmed copy of a result. Guessing a missing
    direction as zero once turned a significant +4.7% win into "significantly worse", so a
    missing field is derived from what is there or rejected, never assumed.
    """
    if not isinstance(result, dict):
        raise ValueError(f"{name} must be a result object from one of the tests")
    result = dict(result)
    
    if result.get('mean_difference') is None:
        control, treatment = result.get('control_mean'), result.get('treatment_mean')
        if control is None or treatment is None:
            raise ValueError(
                f"{name} has no 'mean_difference' and no 'control_mean' / 'treatment_mean' to derive it from. "
                "Pass the test's result object unmodified."
            )
        result['mean_difference'] = treatment - control
    
    if result.get('p_value') is None:
        raise ValueError(f"{name} has no 'p_value'. Pass the test's result object unmodified.")
    if result.get('significant') is None:
        result['significant'] = bool(result['p_value'] < result.get('alpha', 0.05))
    return result


def _harmful(result: Dict, higher_is_better: bool) -> bool:
    """A metric moved significantly in the wrong direction"""
    if not result['significant'] or result['mean_difference'] == 0:
        return False
    return (result['mean_difference'] > 0) != higher_is_better


def ship_decision(
    primary: Dict,
    guardrails: Optional[Dict[str, Dict]] = None,
    health: Optional[Dict] = None,
    higher_is_better: bool = True,
    guardrail_higher_is_better: Optional[Dict[str, bool]] = None,
    mde_pct: Optional[float] = None
) -> Dict:
    """
    Turn test results into a single decision with the reasons behind it
    
    Order of precedence:
      1. Sample ratio mismatch -> results cannot be trusted, nothing else matters
      2. Any guardrail significantly harmed -> do not ship
      3. Primary significantly worse -> do not ship
      4. Primary significantly better -> ship
      5. Not significant, but the interval rules out the minimum effect you
         care about -> stop, more data will not change the answer
      6. Otherwise -> keep running
    
    Args:
        primary: Result dict for the primary metric
        guardrails: {metric name: result dict} for metrics that must not get worse
        health: Output of HealthChecker.run_all_checks
        higher_is_better: Direction of the primary metric
        guardrail_higher_is_better: Direction per guardrail (default: higher is better)
        mde_pct: Smallest relative lift worth shipping, in percent
    """
    primary = _checked(primary, 'primary')
    guardrails = {name: _checked(r, f"guardrail '{name}'") for name, r in (guardrails or {}).items()}
    directions = guardrail_higher_is_better or {}
    reasons: List[str] = []
    
    srm = (health or {}).get('sample_ratio_mismatch', {})
    if srm.get('has_srm'):
        return {
            'decision': INVALID,
            'reasons': [
                f"Sample ratio mismatch (p={srm['p_value']:.2g}). Assignment or logging is broken, "
                "so the comparison is not apples to apples."
            ],
            'harmed_guardrails': []
        }
    
    bad_days = ((health or {}).get('over_time') or {}).get('days_with_srm') or []
    if bad_days:
        return {
            'decision': INVALID,
            'reasons': [
                f"Sample ratio mismatch on {len(bad_days)} day(s) ({', '.join(d['day'] for d in bad_days[:3])}). "
                "Assignment or logging broke on those days. Fix or exclude them before trusting the result."
            ],
            'harmed_guardrails': []
        }
    
    harmed = [name for name, r in guardrails.items() if _harmful(r, directions.get(name, True))]
    for name in harmed:
        r = guardrails[name]
        reasons.append(
            f"Guardrail '{name}' got significantly worse "
            f"({format_change(r)}, p={r['p_value']:.4f})."
        )
    
    change = format_change(primary)
    primary_bad = _harmful(primary, higher_is_better)
    primary_good = bool(primary.get('significant')) and not primary_bad
    
    if harmed or primary_bad:
        if primary_bad:
            reasons.append(f"Primary metric got significantly worse ({change}, p={primary['p_value']:.4f}).")
        elif primary_good:
            reasons.append(f"Primary metric improved ({change}), but not at the cost of a guardrail.")
        decision = DO_NOT_SHIP
    elif primary_good:
        reasons.append(f"Primary metric improved significantly ({change}, p={primary['p_value']:.4f}).")
        if guardrails:
            reasons.append(f"All {len(guardrails)} guardrail metrics held.")
        decision = SHIP
    else:
        lo, hi = primary.get('lift_ci_lower'), primary.get('lift_ci_upper')
        best_case = (hi if higher_is_better else -lo) if lo is not None and hi is not None else None
        if mde_pct is not None and best_case is not None and best_case < mde_pct:
            reasons.append(
                f"No significant effect, and the confidence interval [{lo:+.2f}%, {hi:+.2f}%] "
                f"rules out the {mde_pct:.1f}% lift you care about. More data will not change this."
            )
            decision = STOP_NO_EFFECT
        else:
            reasons.append(
                f"No significant effect yet (p={primary['p_value']:.4f})"
                + (f", and the interval [{lo:+.2f}%, {hi:+.2f}%] still allows a meaningful lift."
                   if lo is not None and hi is not None else ".")
            )
            decision = KEEP_RUNNING
    
    return {'decision': decision, 'reasons': reasons, 'harmed_guardrails': harmed}
