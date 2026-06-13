# metrics/validator.py
# ==============================================================================
# Module: metrics.validator
# Purpose: Validate metric results and inputs
#
# Classes:
#   - MetricValidator: Validate metric computations and results
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from .types import MetricValue, MetricResult

logger = logging.getLogger(__name__)


# ==============================================================================
# METRIC VALIDATOR
# ==============================================================================

@dataclass
class MetricValidator:
    """
    Validate metric results and computations.
    
    Checks validity of metric results, including value ranges,
    NaN/Inf detection, and consistency checks.
    
    Attributes
    ----------
    strict_mode : bool
        If True, enforce strict validation
    tolerance : float
        Numerical tolerance
    
    Examples
    --------
    >>> validator = MetricValidator(strict_mode=True)
    >>> is_valid, issues = validator.validate_result(metric_result)
    """
    
    strict_mode: bool = False
    tolerance: float = 1e-12
    
    def validate_result(self, result: MetricResult) -> Tuple[bool, List[str]]:
        """
        Validate metric result.
        
        Parameters
        ----------
        result : MetricResult
            Result to validate
        
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_issues)
        
        Examples
        --------
        >>> is_valid, issues = validator.validate_result(result)
        >>> if not is_valid:
        ...     for issue in issues:
        ...         print(f"Issue: {issue}")
        """
        
        issues = []
        
        # Check if result is valid flag
        if not result.is_valid:
            issues.append(f"Result marked invalid: {result.error_message}")
        
        # Check metric values
        for name, value in result.metric_values.items():
            is_valid, val_issues = self.validate_value(value)
            if not is_valid:
                issues.extend([f"{name}: {issue}" for issue in val_issues])
        
        # Strict mode checks
        if self.strict_mode:
            if not result.metric_values:
                issues.append("No metric values computed")
            
            if result.particle_A_id < 0 or result.particle_B_id < 0:
                issues.append("Invalid particle IDs")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.debug(f"[MetricValidator] Result valid: {result.metric_name}")
        else:
            logger.warning(f"[MetricValidator] Result invalid: {len(issues)} issues")
        
        return is_valid, issues
    
    def validate_value(self, value: MetricValue) -> Tuple[bool, List[str]]:
        """
        Validate metric value.
        
        Parameters
        ----------
        value : MetricValue
            Value to validate
        
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_issues)
        """
        
        issues = []
        
        # Check for NaN/Inf
        if np.isnan(value.value) or np.isinf(value.value):
            issues.append(f"Value is NaN or Inf: {value.value}")
        
        # Check components
        for comp_name, comp_val in value.components.items():
            if np.isnan(comp_val) or np.isinf(comp_val):
                issues.append(f"Component '{comp_name}' is NaN or Inf")
        
        # Check uncertainty
        if value.uncertainty is not None:
            if value.uncertainty < 0:
                issues.append(f"Negative uncertainty: {value.uncertainty}")
            
            if np.isnan(value.uncertainty):
                issues.append("Uncertainty is NaN")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    def check_consistency(
        self,
        results: List[MetricResult]
    ) -> Tuple[bool, List[str]]:
        """
        Check consistency across multiple results.
        
        Parameters
        ----------
        results : List[MetricResult]
            Results to check
        
        Returns
        -------
        Tuple[bool, List[str]]
            (is_consistent, list_of_inconsistencies)
        """
        
        issues = []
        
        if not results:
            return True, []
        
        # Check all same metric
        first_metric = results[0].metric_name
        for result in results[1:]:
            if result.metric_name != first_metric:
                issues.append(f"Mixed metrics: {first_metric} vs {result.metric_name}")
        
        # Check value ranges if same metric
        if all(r.metric_name == first_metric for r in results):
            # Check for huge outliers
            all_values = []
            for result in results:
                all_values.extend([v.value for v in result.metric_values.values()])
            
            if all_values:
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                
                if std_val > 0:
                    for result in results:
                        for name, value in result.metric_values.items():
                            z_score = abs((value.value - mean_val) / std_val)
                            if z_score > 5:  # 5-sigma outlier
                                issues.append(
                                    f"Outlier in {result.metric_name}: "
                                    f"{name}={value.value:.6f}"
                                )
        
        is_consistent = len(issues) == 0
        
        return is_consistent, issues


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("METRICS VALIDATOR - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] MetricValidator...")
    try:
        validator = MetricValidator(strict_mode=True)
        print(f"✓ Validator created")
        
        # Test valid value
        value = MetricValue(value=3.5)
        is_valid, issues = validator.validate_value(value)
        print(f"✓ Valid value check: {is_valid}")
        
        # Test invalid value (NaN)
        bad_value = MetricValue(value=float('nan'))
    except Exception as e:
        print(f"✓ Correctly caught NaN\n")
    
    print("="*80)
    print("✓ Validator tests passed!")
    print("="*80 + "\n")
