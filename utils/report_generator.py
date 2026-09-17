"""
Report Generator
Exports experiment results to Excel, Markdown, and HTML
"""

import html as html_lib
import re
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import io


RAW_DATA_ROW_LIMIT = 100_000


def _scalar_items(results: Dict):
    """Result entries that fit in a two-column table"""
    for key, value in results.items():
        if isinstance(value, (int, float, str, bool)):
            yield key, value


def _markdown_to_html(text: str) -> str:
    """Convert the small Markdown subset the interpreters emit (headings, bold, bullets)"""
    blocks = []
    for block in html_lib.escape(text, quote=False).split('\n\n'):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', block)
        heading = re.match(r'^(#{2,4})\s+(.*)$', block)
        if heading:
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{heading.group(2)}</h{level}>")
        elif all(line.startswith('- ') for line in block.split('\n')):
            items = ''.join(f"<li>{line[2:]}</li>" for line in block.split('\n'))
            blocks.append(f"<ul>{items}</ul>")
        else:
            blocks.append(f"<p>{block.replace(chr(10), '<br>')}</p>")
    return '\n'.join(blocks)


def _decision_markdown(decision: Optional[Dict]) -> str:
    if not decision:
        return ""
    reasons = '\n'.join(f"- {r}" for r in decision['reasons'])
    return f"## 🚦 Decision: {decision['decision']}\n\n{reasons}\n\n"


class ReportGenerator:
    """Generates exportable reports from experiment results"""
    
    @property
    def timestamp(self) -> str:
        """Fresh on every export, so file names do not repeat within a session"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_excel_report(
        self,
        results: Dict,
        data: pd.DataFrame,
        test_type: str
    ) -> io.BytesIO:
        """
        Create comprehensive Excel report
        
        Args:
            results: Test results dictionary
            data: Original data
            test_type: Type of test performed
            
        Returns:
            BytesIO object with Excel file
        """
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [],
                'Value': []
            }
            
            summary_data['Metric'].append('test_type')
            summary_data['Value'].append(test_type)
            for key, value in _scalar_items(results):
                if key == 'test_type':
                    continue
                summary_data['Metric'].append(key)
                summary_data['Value'].append(value)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Tabular results (balance table, event-study coefficients, per-variant comparisons)
            for key, value in results.items():
                if key in ('matched_treated', 'matched_control'):
                    continue
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    value = pd.DataFrame([dict(_scalar_items(row)) for row in value])
                if isinstance(value, pd.DataFrame):
                    value.to_excel(writer, sheet_name=key[:31], index=False)
            
            # Data sheet, capped: a sheet holds 1,048,576 rows at most and large workbooks exhaust memory
            if len(data) > RAW_DATA_ROW_LIMIT:
                pd.DataFrame({'Note': [
                    f"Raw data omitted: {len(data):,} rows exceeds the {RAW_DATA_ROW_LIMIT:,}-row export limit."
                ]}).to_excel(writer, sheet_name='Raw Data', index=False)
            else:
                data.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Format workbook
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })
            
            for sheet_name in writer.sheets:
                writer.sheets[sheet_name].set_column('A:Z', 18)
            
            # Only the summary sheet has the Metric / Value header
            for col_num, value in enumerate(summary_df.columns.values):
                writer.sheets['Summary'].write(0, col_num, value, header_format)
        
        output.seek(0)
        return output
    
    def create_markdown_report(
        self,
        results: Dict,
        interpretation: str,
        test_type: str,
        decision: Optional[Dict] = None
    ) -> str:
        """
        Create Markdown report
        
        Args:
            results: Test results
            interpretation: Business interpretation
            test_type: Type of test
            
        Returns:
            Markdown string
        """
        report = f"# Experiment Analysis Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Test Type:** {test_type}\n\n"
        report += "---\n\n"
        
        report += _decision_markdown(decision)
        
        # Add interpretation
        report += interpretation + "\n\n"
        
        # Add detailed results
        report += "## Detailed Results\n\n"
        report += "```\n"
        for key, value in _scalar_items(results):
            report += f"{key}: {value}\n"
        report += "```\n\n"
        
        report += "---\n\n"
        report += "*Report generated by Experimentation & Causal Analysis Suite*\n"
        
        return report
    
    def create_html_report(
        self,
        results: Dict,
        interpretation: str,
        test_type: str,
        include_charts: bool = True,
        decision: Optional[Dict] = None
    ) -> str:
        """
        Create HTML report
        
        Args:
            results: Test results
            interpretation: Business interpretation
            test_type: Type of test
            include_charts: Whether to include chart placeholders
            
        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 30px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                .section {{
                    background-color: white;
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                    padding: 15px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                    min-width: 200px;
                }}
                .metric-label {{
                    font-size: 12px;
                    color: #7f8c8d;
                    text-transform: uppercase;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-top: 5px;
                }}
                .interpretation {{
                    line-height: 1.8;
                    color: #34495e;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                .footer {{
                    text-align: center;
                    color: #7f8c8d;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Experiment Analysis Report</h1>
                <p><strong>Test Type:</strong> {html_lib.escape(str(test_type))}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
        """
        
        # Add key metrics
        key_metrics = [
            ('p_value', 'P-Value'),
            ('control_mean', 'Control Mean'),
            ('treatment_mean', 'Treatment Mean'),
            ('relative_lift', 'Relative Lift (%)'),
            ('control_n', 'Control Sample Size'),
            ('treatment_n', 'Treatment Sample Size')
        ]
        
        for key, label in key_metrics:
            if key in results:
                value = results[key]
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                html += f"""
                <div class="metric">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
                """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Business Interpretation</h2>
                <div class="interpretation">
        """
        
        html += _markdown_to_html(_decision_markdown(decision) + interpretation)
        
        html += """
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for key, value in _scalar_items(results):
            html += f"""
                        <tr>
                            <td>{html_lib.escape(str(key))}</td>
                            <td>{html_lib.escape(str(value))}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Report generated by Experimentation & Causal Analysis Suite</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def export_results_summary(
        self,
        results: Dict
    ) -> pd.DataFrame:
        """
        Create a summary DataFrame from results
        
        Args:
            results: Results dictionary
            
        Returns:
            DataFrame with summary
        """
        return pd.DataFrame(
            [{'Metric': key, 'Value': value} for key, value in _scalar_items(results)]
        )
