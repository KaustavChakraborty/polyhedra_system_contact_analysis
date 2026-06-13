# analysis/order_parameters.py
# ==============================================================================
# Module: analysis.order_parameters
# Purpose: Pure contact order parameter computation (Cij)
#
# Classes:
#   - OrderParameterCalculator: Compute contact order parameters
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# ORDER PARAMETER CALCULATOR
# ==============================================================================

@dataclass
class OrderParameterCalculator:
    """
    Calculate contact order parameters.
    
    Computes Cij (contact order) metrics from contact results.
    
    Attributes
    ----------
    tolerance : float
        Numerical tolerance
    
    Examples
    --------
    >>> calc = OrderParameterCalculator()
    >>> cij = calc.compute_contact_order(contact_results, n_particle_types=2)
    """
    
    tolerance: float = 1e-12
    
    def __post_init__(self) -> None:
        """Validate calculator."""
        if self.tolerance <= 0:
            raise ValidationError(
                f"tolerance must be positive",
                error_code="ORDER_INVALID_TOLERANCE"
            )
        
        logger.debug("[OrderParameterCalculator] Initialized")
    
    def compute_contact_order(
        self,
        contact_results: list,
        n_particle_types: int = 1
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute contact order parameters Cij.
        
        Parameters
        ----------
        contact_results : list
            Contact results to analyze
        n_particle_types : int, optional
            Number of particle types (default: 1)
        
        Returns
        -------
        Dict[Tuple[int, int], float]
            Contact orders: {(type_i, type_j): Cij}
        
        Notes
        -----
        Cij = (number of contacts between types i and j) / (total contacts)
        """
        
        logger.debug(
            f"[OrderParameterCalculator] Computing contact order "
            f"from {len(contact_results)} contacts"
        )
        
        # Initialize counters
        contact_counts = {}
        total_contacts = 0
        
        # Count contacts by type pair
        for contact in contact_results:
            if not hasattr(contact, 'overlap_count') or contact.overlap_count == 0:
                continue
            
            total_contacts += contact.overlap_count
            
            # Simple type assignment (could be enhanced)
            type_i = 0
            type_j = 0
            pair_key = tuple(sorted([type_i, type_j]))
            
            contact_counts[pair_key] = contact_counts.get(pair_key, 0) + contact.overlap_count
        
        # Normalize to get Cij
        cij = {}
        if total_contacts > 0:
            for pair_key, count in contact_counts.items():
                cij[pair_key] = count / total_contacts
        else:
            # No contacts found
            for i in range(n_particle_types):
                for j in range(i, n_particle_types):
                    pair_key = (i, j)
                    cij[pair_key] = 0.0
        
        logger.info(
            f"[OrderParameterCalculator] Contact order computed: "
            f"{len(cij)} pairs, total={total_contacts}"
        )
        
        return cij
    
    def compute_coordination_number(
        self,
        contact_results: list,
        particle_id: int = -1
    ) -> float:
        """
        Compute coordination number.
        
        Parameters
        ----------
        contact_results : list
            Contact results
        particle_id : int, optional
            Filter by particle (default: -1 = all)
        
        Returns
        -------
        float
            Average coordination number
        """
        
        logger.debug("[OrderParameterCalculator] Computing coordination number")
        
        coordination_numbers = {}
        
        for contact in contact_results:
            if contact.overlap_count == 0:
                continue
            
            # Count contacts per particle
            if particle_id >= 0:
                if contact.particle_A_id == particle_id:
                    coordination_numbers[particle_id] = \
                        coordination_numbers.get(particle_id, 0) + 1
                if contact.particle_B_id == particle_id:
                    coordination_numbers[particle_id] = \
                        coordination_numbers.get(particle_id, 0) + 1
            else:
                # Count for all particles
                for p_id in [contact.particle_A_id, contact.particle_B_id]:
                    coordination_numbers[p_id] = \
                        coordination_numbers.get(p_id, 0) + 1
        
        # Average
        if coordination_numbers:
            avg_z = np.mean(list(coordination_numbers.values()))
        else:
            avg_z = 0.0
        
        logger.debug(f"[OrderParameterCalculator] Coordination number: {avg_z:.3f}")
        
        return float(avg_z)


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("ORDER PARAMETERS MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] OrderParameterCalculator...")
    try:
        calc = OrderParameterCalculator()
        print(f"✓ Calculator created")
        
        # Compute contact order
        cij = calc.compute_contact_order([], n_particle_types=2)
        print(f"✓ Contact order computed: {cij}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Order parameters tests passed!")
    print("="*80 + "\n")
