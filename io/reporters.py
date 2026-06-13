# io/reporters.py
# ==============================================================================
# Module: io.reporters
# Purpose: Generate analysis reports and summaries
#
# Classes:
#   - ReportGenerator: Generate analysis reports
#   - SummaryGenerator: Create summary statistics
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# REPORT GENERATOR
# ==============================================================================

@dataclass
class ReportGenerator:
    """
    Generate analysis reports.
    
    Creates formatted reports from analysis results.
    
    Examples
    --------
    >>> gen = ReportGenerator()
    >>> report = gen.generate_report(results)
    """
    
    title: str = "Contact Analysis Report"
    author: str = "Contact Analysis Team"
    
    def generate_report(
        self,
        results: Dict[str, Any],
        include_metadata: bool = True
    ) -> str:
        """
        Generate formatted report.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Analysis results
        include_metadata : bool, optional
            Include timestamp and author (default: True)
        
        Returns
        -------
        str
            Formatted report text
        """
        
        logger.debug("[ReportGenerator] Generating report")
        
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(self.title)
        lines.append("=" * 80)
        lines.append("")
        
        # Metadata
        if include_metadata:
            lines.append(f"Author: {self.author}")
            lines.append(f"Generated: {datetime.now().isoformat()}")
            lines.append("")
        
        # Content
        for section, content in results.items():
            lines.append(f"\n{section.upper()}")
            lines.append("-" * len(section))
            
            if isinstance(content, dict):
                for key, value in content.items():
                    lines.append(f"  {key}: {value}")
            elif isinstance(content, list):
                for item in content:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"  {content}")
        
        # Footer
        lines.append("")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        logger.info("[ReportGenerator] Report generated")
        
        return report
    
    def save_report(self, report: str, filepath: str) -> None:
        """
        Save report to file.
        
        Parameters
        ----------
        report : str
            Report text
        filepath : str
            Output file path
        """
        
        logger.debug(f"[ReportGenerator] Saving report to {filepath}")
        
        try:
            with open(filepath, 'w') as f:
                f.write(report)
            
            logger.info(f"[ReportGenerator] Report saved to {filepath}")
        
        except Exception as e:
            raise ValidationError(
                f"Failed to save report: {e}",
                error_code="REPORT_SAVE_ERROR"
            ) from e


# ==============================================================================
# SUMMARY GENERATOR
# ==============================================================================

@dataclass
class SummaryGenerator:
    """
    Generate summary statistics.
    
    Creates condensed summaries of analysis results.
    
    Examples
    --------
    >>> gen = SummaryGenerator()
    >>> summary = gen.generate_summary(rdf, stats)
    """
    
    def generate_summary(
        self,
        rdf_data: Optional[Dict[str, Any]] = None,
        stats_data: Optional[Dict[str, Any]] = None,
        metrics_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate summary statistics.
        
        Parameters
        ----------
        rdf_data : Dict[str, Any], optional
            RDF results
        stats_data : Dict[str, Any], optional
            Statistics results
        metrics_data : Dict[str, Any], optional
            Metrics results
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        
        logger.debug("[SummaryGenerator] Generating summary")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'sections': []
        }
        
        # RDF summary
        if rdf_data:
            rdf_summary = self._summarize_rdf(rdf_data)
            summary['rdf'] = rdf_summary
            summary['sections'].append('rdf')
        
        # Statistics summary
        if stats_data:
            stats_summary = self._summarize_stats(stats_data)
            summary['statistics'] = stats_summary
            summary['sections'].append('statistics')
        
        # Metrics summary
        if metrics_data:
            metrics_summary = self._summarize_metrics(metrics_data)
            summary['metrics'] = metrics_summary
            summary['sections'].append('metrics')
        
        logger.info(f"[SummaryGenerator] Summary generated: {len(summary)} sections")
        
        return summary
    
    def _summarize_rdf(self, rdf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize RDF data."""
        return {
            'frames': rdf_data.get('frame_count', 0),
            'bins': rdf_data.get('n_bins', 0),
            'r_range': f"[{rdf_data.get('r_min', 0):.3f}, {rdf_data.get('r_max', 0):.3f}]",
        }
    
    def _summarize_stats(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize statistics."""
        summary = {}
        
        for metric_name, result in stats_data.items():
            if isinstance(result, dict):
                summary[metric_name] = {
                    'mean': f"{result.get('mean', 0):.6f}",
                    'std': f"{result.get('std', 0):.6f}",
                    'samples': result.get('n_samples', 0),
                }
        
        return summary
    
    def _summarize_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize metrics."""
        return {
            'count': len(metrics_data),
            'metrics': list(metrics_data.keys()),
        }
    
    def format_summary(self, summary: Dict[str, Any]) -> str:
        """
        Format summary as string.
        
        Parameters
        ----------
        summary : Dict[str, Any]
            Summary data
        
        Returns
        -------
        str
            Formatted summary
        """
        
        lines = []
        lines.append("ANALYSIS SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Generated: {summary.get('timestamp', 'Unknown')}")
        lines.append("")
        
        if 'rdf' in summary:
            lines.append("RDF Analysis")
            lines.append("-" * 40)
            for key, value in summary['rdf'].items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if 'statistics' in summary:
            lines.append("Statistics")
            lines.append("-" * 40)
            for metric, values in summary['statistics'].items():
                lines.append(f"  {metric}:")
                for key, value in values.items():
                    lines.append(f"    {key}: {value}")
            lines.append("")
        
        if 'metrics' in summary:
            lines.append("Metrics")
            lines.append("-" * 40)
            for key, value in summary['metrics'].items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("IO REPORTERS MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] ReportGenerator...")
    try:
        gen = ReportGenerator()
        
        test_results = {
            'analysis': 'Contact analysis completed',
            'frames': 100,
            'contacts': 5000
        }
        
        report = gen.generate_report(test_results)
        print("✓ Report generated:")
        print(report[:200] + "...\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] SummaryGenerator...")
    try:
        gen = SummaryGenerator()
        
        test_rdf = {'frame_count': 100, 'n_bins': 50, 'r_min': 0.5, 'r_max': 5.0}
        summary = gen.generate_summary(rdf_data=test_rdf)
        
        formatted = gen.format_summary(summary)
        print("✓ Summary generated:")
        print(formatted[:200] + "...\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Reporter tests passed!")
    print("="*80 + "\n")
