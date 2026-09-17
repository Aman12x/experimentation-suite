"""
Statistical Interpreters
Provides plain English explanations of statistical results
"""

from typing import Dict, List
import numpy as np


class StatisticalInterpreter:
    """Translates statistical jargon into business-friendly language"""
    
    @staticmethod
    def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
        """
        Explain what a p-value means in plain English
        
        Args:
            p_value: The p-value from a test
            alpha: Significance level
            
        Returns:
            Plain English explanation
        """
        if p_value < alpha:
            strength = "strong" if p_value < 0.01 else "moderate"
            return (
                f"📊 **Statistically Significant** (p={p_value:.4f})\n\n"
                f"We have {strength} evidence of a real difference between the groups. "
                f"If there were truly no difference, a result at least this extreme "
                f"would show up only {p_value*100:.2f}% of the time."
            )
        else:
            return (
                f"📊 **Not Statistically Significant** (p={p_value:.4f})\n\n"
                f"We don't have sufficient evidence to conclude there's a real difference. "
                f"If there were truly no difference, a result at least this extreme would still "
                f"show up {p_value*100:.1f}% of the time, so we cannot confidently say the treatment had an effect."
            )
    
    @staticmethod
    def interpret_confidence_interval(
        ci_lower: float,
        ci_upper: float,
        metric_name: str = "metric",
        confidence_level: float = 0.95
    ) -> str:
        """
        Explain confidence interval in plain English
        
        Args:
            ci_lower: Lower bound of CI
            ci_upper: Upper bound of CI
            metric_name: Name of the metric
            confidence_level: Confidence level
            
        Returns:
            Plain English explanation
        """
        contains_zero = ci_lower <= 0 <= ci_upper
        
        if contains_zero:
            return (
                f"📏 **{int(confidence_level*100)}% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
                f"We're {int(confidence_level*100)}% confident the true effect on {metric_name} "
                f"falls within this range. **Since this range includes zero**, we cannot rule out "
                f"the possibility of no effect. The treatment might increase, decrease, or have no impact on the metric."
            )
        elif ci_lower > 0:
            return (
                f"📏 **{int(confidence_level*100)}% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
                f"We're {int(confidence_level*100)}% confident the true effect on {metric_name} "
                f"is somewhere between {ci_lower:.4f} and {ci_upper:.4f}. "
                f"**Since both bounds are positive**, we can be confident the treatment has a positive effect."
            )
        else:
            return (
                f"📏 **{int(confidence_level*100)}% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
                f"We're {int(confidence_level*100)}% confident the true effect on {metric_name} "
                f"is somewhere between {ci_lower:.4f} and {ci_upper:.4f}. "
                f"**Since both bounds are negative**, we can be confident the treatment has a negative effect."
            )
    
    @staticmethod
    def interpret_effect_size(cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size
        
        Args:
            cohens_d: Cohen's d value
            
        Returns:
            Plain English interpretation
        """
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        direction = "increase" if cohens_d > 0 else "decrease"
        
        return (
            f"📐 **Effect Size (Cohen's d):** {cohens_d:.3f}\n\n"
            f"This represents a **{magnitude}** {direction} in the metric. "
            f"Cohen's d measures how many standard deviations separate the two groups. "
            f"A value of {abs_d:.2f} means the difference is {magnitude} in practical terms."
        )
    
    @staticmethod
    def interpret_ab_test_results(results: Dict) -> str:
        """
        Provide comprehensive business interpretation of A/B test
        
        Args:
            results: Dictionary with test results
            
        Returns:
            Plain English business interpretation
        """
        test_type = results.get('test_type', 'Statistical Test')
        p_value = results['p_value']
        significant = results['significant']
        
        # Get means
        control_mean = results.get('control_mean', 0)
        treatment_mean = results.get('treatment_mean', 0)
        relative_lift = results.get('relative_lift', 0)
        
        # Build interpretation
        interpretation = f"## 🎯 Business Interpretation ({test_type})\n\n"
        
        # Main result
        if significant:
            direction = "higher" if treatment_mean > control_mean else "lower"
            interpretation += (
                f"### ✅ Significant Result Detected\n\n"
                f"The treatment group performed **{direction}** than the control group "
                f"with **{abs(relative_lift):.2f}% {direction.replace('higher', 'increase').replace('lower', 'decrease')}**.\n\n"
            )
            
            if abs(relative_lift) < 1:
                interpretation += (
                    f"⚠️ **Note:** While statistically significant, the effect is quite small "
                    f"({abs(relative_lift):.2f}%). Consider whether this improvement justifies "
                    f"the implementation cost.\n\n"
                )
            elif relative_lift > 20:
                interpretation += (
                    f"🚀 **Strong Impact:** This is a substantial effect ({abs(relative_lift):.2f}%). "
                    f"Strong candidate for implementation.\n\n"
                )
            elif relative_lift < -20:
                interpretation += (
                    f"🛑 **Strong Negative Impact:** The treatment lowered the metric by "
                    f"{abs(relative_lift):.2f}%.\n\n"
                )
        else:
            interpretation += (
                f"### ❌ No Significant Difference Found\n\n"
                f"The data doesn't provide strong enough evidence that the treatment "
                f"actually changed the metric. A {abs(relative_lift):.2f}% difference "
                f"is within what random variation alone can produce at this sample size.\n\n"
            )
        
        # Statistical details
        interpretation += f"### 📊 Statistical Details\n\n"
        interpretation += f"- **Control Mean:** {control_mean:.4f}\n"
        interpretation += f"- **Treatment Mean:** {treatment_mean:.4f}\n"
        interpretation += f"- **Relative Change:** {relative_lift:+.2f}%\n"
        interpretation += f"- **P-value:** {p_value:.4f}\n"
        interpretation += f"- **Sample Sizes:** Control={results.get('control_n', 'N/A')}, Treatment={results.get('treatment_n', 'N/A')}\n\n"
        
        # Confidence interval interpretation
        if 'ci_lower' in results and 'ci_upper' in results:
            interpretation += StatisticalInterpreter.interpret_confidence_interval(
                results['ci_lower'],
                results['ci_upper'],
                "the metric"
            ) + "\n\n"
        
        # Effect size interpretation
        if 'cohens_d' in results:
            interpretation += StatisticalInterpreter.interpret_effect_size(
                results['cohens_d']
            ) + "\n\n"
        
        # Recommendation
        interpretation += "### 💡 Recommendation\n\n"
        if significant and relative_lift < 0:
            interpretation += (
                "**❌ DO NOT IMPLEMENT:** The treatment performed significantly worse than control. "
                "If a lower value of this metric is the goal (e.g. churn or load time), read this as a win instead."
            )
        elif significant and relative_lift > 2:
            interpretation += (
                "**✅ RECOMMEND:** Implement the treatment. The results show a statistically "
                "significant and practically meaningful improvement."
            )
        elif significant:
            interpretation += (
                "**⚠️ PROCEED WITH CAUTION:** While statistically significant, the effect size is small. "
                "Weigh the implementation costs against the modest gains."
            )
        else:
            interpretation += (
                "**❌ DO NOT IMPLEMENT:** There's insufficient evidence that the treatment "
                "provides a benefit. Consider running a larger test or trying alternative approaches."
            )
        
        return interpretation
    
    @staticmethod
    def interpret_power_analysis(power_results: Dict) -> str:
        """
        Interpret power analysis results
        
        Args:
            power_results: Dictionary with power analysis results
            
        Returns:
            Plain English interpretation
        """
        total_n = power_results.get('total_sample_size', 0)
        mde = power_results.get('mde_percentage', 0)
        power = power_results.get('power', 0.8)
        
        return (
            f"## 📏 Sample Size Recommendation\n\n"
            f"To detect a **{mde:.1f}% change** with **{int(power*100)}% power**:\n\n"
            f"- **Total Sample Size Needed:** {total_n:,}\n"
            f"- **Per Group:** ~{total_n//2:,} in control, ~{total_n//2:,} in treatment\n\n"
            f"### What does this mean?\n\n"
            f"With this sample size, you'll have an {int(power*100)}% chance of detecting "
            f"a {mde:.1f}% effect if it truly exists. Running with fewer participants "
            f"increases the risk of missing a real effect (false negative)."
        )
    
    @staticmethod
    def interpret_bayesian_results(results: Dict) -> str:
        """
        Interpret Bayesian A/B test results
        
        Args:
            results: Dictionary with Bayesian results
            
        Returns:
            Plain English interpretation
        """
        prob_better = results['prob_treatment_better']
        expected_lift = results['expected_lift']
        
        interpretation = f"## 🎲 Bayesian Analysis\n\n"
        
        if prob_better > 0.95:
            interpretation += (
                f"### ✅ High Confidence in Treatment\n\n"
                f"There's a **{prob_better*100:.1f}% probability** that the treatment "
                f"performs better than control. This is very strong evidence in favor of the treatment.\n\n"
            )
        elif prob_better > 0.90:
            interpretation += (
                f"### ✅ Good Confidence in Treatment\n\n"
                f"There's a **{prob_better*100:.1f}% probability** that the treatment "
                f"performs better than control. This is good evidence in favor of the treatment.\n\n"
            )
        elif prob_better < 0.10:
            interpretation += (
                f"### ❌ High Confidence Treatment is Worse\n\n"
                f"There's only a **{prob_better*100:.1f}% probability** that the treatment "
                f"performs better (meaning {(1-prob_better)*100:.1f}% it's worse). "
                f"Strong evidence against implementing the treatment.\n\n"
            )
        else:
            interpretation += (
                f"### ⚠️ Uncertain Results\n\n"
                f"There's a **{prob_better*100:.1f}% probability** that the treatment "
                f"performs better. This is not strong enough evidence either way. "
                f"Consider collecting more data.\n\n"
            )
        
        interpretation += (
            f"**Expected Lift:** {expected_lift:+.2f}%\n\n"
            f"95% Credible Interval: [{results['lift_ci_lower']:.2f}%, {results['lift_ci_upper']:.2f}%]\n\n"
            f"This means we're 95% certain the true lift falls within this range."
        )
        
        return interpretation
    
    @staticmethod
    def interpret_causal_effect(results: Dict, method: str) -> str:
        """
        Interpret causal inference results
        
        Args:
            results: Results from causal inference method
            method: Method name (PSM, DiD, etc.)
            
        Returns:
            Plain English interpretation
        """
        if method == "PSM":
            att = results['att']
            match_rate = results.get('match_rate', 100)
            
            interpretation = (
                f"## 🎯 Propensity Score Matching Results\n\n"
                f"### Average Treatment Effect on the Treated (ATT)\n\n"
                f"**Effect:** {att:+.4f}\n\n"
                f"This is the estimated causal effect of the treatment after accounting for "
                f"observable differences between groups through matching.\n\n"
            )
            
            if match_rate < 80:
                interpretation += (
                    f"⚠️ **Match Rate:** Only {match_rate:.1f}% of treated units were matched. "
                    f"This suggests substantial differences between groups. Results may not "
                    f"generalize to all treated units.\n\n"
                )
            
            if results.get('p_value', 1) < 0.05:
                interpretation += (
                    f"✅ **Statistically Significant:** The treatment effect is unlikely to be "
                    f"due to chance (p={results['p_value']:.4f}).\n\n"
                )
            
        elif method == "DiD":
            did_estimate = results['did_estimate']
            
            interpretation = (
                f"## 📊 Difference-in-Differences Results\n\n"
                f"### Treatment Effect Estimate\n\n"
                f"**DiD Estimate:** {did_estimate:+.4f}\n\n"
                f"This is the estimated causal effect of the intervention, controlling for "
                f"pre-existing differences between groups and common time trends.\n\n"
            )
            
            if results.get('p_value', 1) < 0.05:
                interpretation += (
                    f"✅ **Statistically Significant:** p={results['p_value']:.4f}, "
                    f"95% CI [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}] "
                    f"using {results.get('se_type', 'OLS')} standard errors.\n\n"
                )
            
            parallel = results.get('parallel_trends_assumption')
            if parallel is None:
                interpretation += (
                    f"ℹ️ **Parallel Trends:** Not testable with one pre and one post period. "
                    f"The estimate is only causal if both groups would have moved together without "
                    f"the intervention. A gap in levels between groups is fine; diverging trends are not. "
                    f"Check it with several pre-treatment periods if you have them.\n\n"
                )
            elif parallel:
                interpretation += (
                    f"✅ **Parallel Trends:** The pre-treatment trends appear similar, "
                    f"supporting the validity of the DiD approach.\n\n"
                )
            else:
                interpretation += (
                    f"⚠️ **Warning:** Pre-treatment trends may differ, which could violate "
                    f"the parallel trends assumption required for DiD.\n\n"
                )
        
        elif method == "IV":
            interpretation = (
                f"## 🎻 Instrumental Variables Results\n\n"
                f"**2SLS Estimate:** {results['iv_estimate']:+.4f} "
                f"(95% CI [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}], p={results['p_value']:.4f})\n\n"
                f"**Naive OLS Estimate:** {results['ols_estimate']:+.4f}. A large gap between the two "
                f"points to confounding that the instrument removes.\n\n"
            )
            
            if results.get('weak_instrument'):
                interpretation += (
                    f"⚠️ **Weak Instrument:** The instrument's first-stage F is "
                    f"{results['first_stage_f_stat']:.2f} (< 10). 2SLS estimates are biased toward OLS "
                    f"and the confidence interval is unreliable.\n\n"
                )
            else:
                interpretation += (
                    f"✅ **Instrument Strength:** First-stage F for the instrument is "
                    f"{results['first_stage_f_stat']:.2f} (≥ 10).\n\n"
                )
        
        else:
            raise ValueError(f"Unknown method '{method}'")
        
        return interpretation
