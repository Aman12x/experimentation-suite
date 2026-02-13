# 🚀 Quick Start Guide

## Installation (5 minutes)

1. **Extract the files** to a directory of your choice

2. **Open terminal/command prompt** in that directory

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Launch the app**:
```bash
streamlit run app.py
```

5. **Open browser** to http://localhost:8501

## First Analysis (2 minutes)

### Try the Sample A/B Test Data

1. **Upload Data**:
   - Click "Browse files" in the sidebar
   - Select `sample_ab_test_data.csv`
   - Wait for "✅ Loaded" message

2. **Run A/B Test**:
   - Go to "🧪 A/B Testing" tab
   - Select:
     - Group Column: `group`
     - Metric Column: `revenue`
     - Test Type: `T-Test`
   - Click "🚀 Run A/B Test"

3. **Review Results**:
   - Check health diagnostics (should pass ✅)
   - Read the business interpretation
   - Explore interactive charts
   - Note the ~11% revenue lift

### Try Difference-in-Differences

1. **Upload** `sample_did_data.csv`

2. **Go to** "🎯 Causal Inference" tab

3. **Select** "Difference-in-Differences (DiD)"

4. **Configure**:
   - Group Column: `region`
   - Time Period Column: `period`
   - Outcome Column: `sales`
   - Treatment Group: `treatment`
   - Post-Treatment Period: `post`

5. **Click** "🚀 Run DiD Analysis"

6. **Observe** the ~200 unit treatment effect with parallel trends visualization

## Key Features to Explore

### 📏 Power Analysis
In the A/B Testing tab:
- Scroll to "Sample Size Calculator"
- Enter expected parameters
- Get sample size recommendations

### 🎯 Propensity Score Matching
In Causal Inference tab:
- Select PSM method
- Choose treatment, outcome, and covariates
- Review balance diagnostics

### 📄 Export Reports
In Report Export tab:
- Choose your analysis results
- Select format (Excel/Markdown/HTML)
- Download professional report

## Understanding Results

### Statistical Significance
- **p < 0.05**: Strong evidence of real effect
- **p ≥ 0.05**: Insufficient evidence, could be chance

### Effect Size (Cohen's d)
- **< 0.2**: Negligible
- **0.2-0.5**: Small
- **0.5-0.8**: Medium
- **> 0.8**: Large

### Sample Ratio Mismatch
- **OK**: Traffic split as expected
- **CRITICAL**: Data quality issue - investigate!

### Confidence Intervals
- **Excludes 0**: Confident in direction of effect
- **Includes 0**: Effect could be positive, negative, or none

## Common Issues

### "Module not found" error
```bash
pip install --upgrade -r requirements.txt
```

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

### Charts not displaying
- Ensure you have a modern browser (Chrome, Firefox, Edge)
- Check JavaScript is enabled

## Next Steps

1. **Upload your own data** (CSV or Parquet)
2. **Experiment with different tests** and parameters
3. **Compare results** across methods
4. **Export reports** for stakeholders
5. **Read README.md** for detailed documentation

## Tips for Success

✅ **Always check health diagnostics** before interpreting results
✅ **Use power analysis** to plan experiments
✅ **Consider practical significance** not just statistical
✅ **Document assumptions** in your reports
✅ **Validate with multiple methods** when possible

---

**Need help?** Check the "ℹ️ Quick Help" section in the sidebar or review README.md for comprehensive documentation.
