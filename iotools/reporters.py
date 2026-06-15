# iotools/reporters.py
# ==============================================================================
# Module: iotools.reporters
# Purpose: Generate analysis reports and summaries
#
# This module is the human-readable REPORTING layer of the project.
#
# It does not compute contact geometry. It does not read trajectory files. It does
# not calculate overlap areas or distance metrics. Instead, it takes already-
# computed dictionaries and turns them into:
#
#   1. formatted text reports
#   2. compact summary dictionaries
#
# Typical control flow:
#
#     analysis code computes results
#         |
#         v
#     SummaryGenerator.generate_summary(...)
#         |
#         v
#     ReportGenerator.generate_report(...)
#         |
#         v
#     FileWriter / ReportGenerator.save_report writes output
#
# In short:
#     analysis modules produce numbers;
#     reporters.py explains those numbers in a readable structure.
#
# Classes:
#   - ReportGenerator: Generate analysis reports
#   - SummaryGenerator: Create summary statistics
#
# Author: Kaustav Chakraborty
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

import sys
from pathlib import Path

# ------------------------------------------------------------------------------
# Local project import setup
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# REPORT GENERATOR
# ==============================================================================

@dataclass
class ReportGenerator:
    """
    Convert a result dictionary into a formatted text report.

    Example input
    -------------
    results = {
        "rdf": {"frames": 100, "bins": 50},
        "statistics": {"mean_distance": 0.42},
        "metrics": ["face_center_face_center", "vertex_vertex"],
    }

    Example output
    --------------
    A multi-line string with section headers and key/value pairs.
    """
    
    title: str = "Contact Analysis Report"
    author: str = "Kaustav Chakraborty"
    
    def generate_report(self, results: Dict[str, Any], include_metadata: bool = True) -> str:
        """
        Generate a formatted report string.

        Parameters
        ----------
        results
            Dictionary containing analysis outputs. Each top-level key becomes a
            report section.

        include_metadata
            If True, include author and timestamp at the beginning.

        Returns
        -------
        str
            Complete report as one multi-line string.

        Control flow
        ------------
        1. Validate that results is a dictionary.
        2. Create report header.
        3. Add metadata if requested.
        4. Loop over each result section.
        5. Format nested dictionaries/lists/scalars.
        6. Return the joined text.
        """
        
        logger.debug("[ReportGenerator] Generating report")
        
        if results is None:
            raise ValidationError(
                "ReportGenerator received results=None.",
                error_code="REPORT_RESULTS_NONE",
            )

        if not isinstance(results, dict):
            raise ValidationError(
                "ReportGenerator expects results to be a dictionary.",
                error_code="REPORT_RESULTS_INVALID_TYPE",
            )

        lines: List[str] = []
        
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
        
        # Main content block
        for section, content in results.items():
            section_name = str(section)
            lines.append("")
            lines.append(section_name.upper())
            lines.append("-" * len(section_name))
            self._append_formatted_content(lines, content, indent=2)

        # Footer block
        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)
        logger.info("[ReportGenerator] Report generated")
        return report
    

    def _append_formatted_content(self, lines: List[str], content: Any, indent: int = 2) -> None:
        """
        Append formatted content into an existing report line list.

        This helper recursively handles nested dictionaries. It keeps the main
        generate_report() method easier to read.
        """
        prefix = " " * indent

        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    self._append_formatted_content(lines, value, indent=indent + 2)
                elif isinstance(value, list):
                    lines.append(f"{prefix}{key}:")
                    for item in value:
                        lines.append(f"{prefix}  - {item}")
                else:
                    lines.append(f"{prefix}{key}: {value}")

        elif isinstance(content, list):
            for item in content:
                lines.append(f"{prefix}- {item}")

        else:
            lines.append(f"{prefix}{content}")



    def save_report(self, report: str, filepath: str | Path) -> None:
        """
        Save report text to disk.

        This method is convenient if you already have a report string and want
        to write it directly without going through FileWriter.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug("[ReportGenerator] Saving report to %s", path)

        try:
            with path.open("w", encoding="utf-8") as handle:
                handle.write(report)
            logger.info("[ReportGenerator] Report saved to %s", path)

        except Exception as exc:
            raise ValidationError(
                f"Failed to save report: {exc}",
                error_code="REPORT_SAVE_ERROR",
            ) from exc


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
    
    def generate_summary(self,
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
            "frames": rdf_data.get("frame_count", 0),
            "bins": rdf_data.get("n_bins", 0),
            "r_range": f"[{float(rdf_data.get('r_min', 0)):.3f}, {float(rdf_data.get('r_max', 0)):.3f}]",
        }
    
    def _summarize_stats(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize dictionary-valued statistics entries.

        Expected structure:

            {
                "distance": {"mean": 0.2, "std": 0.05, "n_samples": 1000},
                "area": {"mean": 1.1, "std": 0.10, "n_samples": 1000},
            }
        """        
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
        
        if "statistics" in summary:
            lines.append("Statistics")
            lines.append("-" * 40)
            for metric, values in summary["statistics"].items():
                lines.append(f"  {metric}:")
                if isinstance(values, dict):
                    for key, value in values.items():
                        lines.append(f"    {key}: {value}")
                else:
                    lines.append(f"    {values}")
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
        print("[SUCCESS] Report generated:")
        print(report[:200] + "...\n")
    except Exception as e:
        print(f"[FAILED] Error: {e}\n")
    
    print("[TEST 2] SummaryGenerator...")
    try:
        gen = SummaryGenerator()
        
        test_rdf = {'frame_count': 100, 'n_bins': 50, 'r_min': 0.5, 'r_max': 5.0}
        summary = gen.generate_summary(rdf_data=test_rdf)
        
        formatted = gen.format_summary(summary)
        print("[SUCCESS] Summary generated:")
        print(formatted[:200] + "...\n")
    except Exception as e:
        print(f"[FAILED] Error: {e}\n")
    
    print("="*80)
    print("[SUCCESS] Reporter tests passed!")
    print("="*80 + "\n")
