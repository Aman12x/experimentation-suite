"""
Ship Decision
Combines the primary metric, guardrail metrics, and health checks into one call
"""

from typing import Dict, List, Optional


SHIP = "SHIP"
DO_NOT_SHIP = "DO NOT SHIP"
KEEP_RUNNING = "KEEP RUNNING"
STOP_NO_EFFECT = "STOP - NO MEANINGFUL EFFECT"
INVALID = "INVALID - FIX THE EXPERIMENT"


def _harmful(result: Dict, higher_is_better: bool) -> bool:
    """A metric moved significantly in the wrong direction"""
    if not result.get('significant'):
        return False
    moved_up = result.get('mean_difference', 0) > 0
    return moved_up != higher_is_better


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
    guardrails = guardrails or {}
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
    
    harmed = [name for name, r in guardrails.items() if _harmful(r, directions.get(name, True))]
    for name in harmed:
        r = guardrails[name]
        reasons.append(
            f"Guardrail '{name}' got significantly worse "
            f"({r.get('relative_lift', 0):+.2f}%, p={r['p_value']:.4f})."
        )
    
    lift = primary.get('relative_lift', 0)
    primary_bad = _harmful(primary, higher_is_better)
    primary_good = bool(primary.get('significant')) and not primary_bad
    
    if harmed or primary_bad:
        if primary_bad:
            reasons.append(f"Primary metric got significantly worse ({lift:+.2f}%, p={primary['p_value']:.4f}).")
        elif primary_good:
            reasons.append(f"Primary metric improved ({lift:+.2f}%), but not at the cost of a guardrail.")
        decision = DO_NOT_SHIP
    elif primary_good:
        reasons.append(f"Primary metric improved significantly ({lift:+.2f}%, p={primary['p_value']:.4f}).")
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
