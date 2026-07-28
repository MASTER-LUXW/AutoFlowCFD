"""Convergence analysis and report generation module.

This module provides tools for analyzing convergence history,
exporting convergence curves to CSV, and generating simulation reports in JSON format.

Key Components:
    - ConvergenceAnalyzer: Residual and coefficient history analysis
    - SimulationReport: Comprehensive simulation report generation

Example:
    >>> from autoflowcfd.postprocess import ConvergenceAnalyzer
    >>> analyzer = ConvergenceAnalyzer()
    >>> analyzer.add_iteration(residuals={'continuity': 1e-3, 'momentum': 1e-4})
    >>> analyzer.export_csv("convergence.csv")
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field, asdict

from .coefficients import AerodynamicCoefficients


@dataclass
class IterationData:
    """Single iteration data
    
    Attributes:
        iteration: Iteration number
        residuals: Residual values for each equation
        cfl: CFL number
        coefficients: Aerodynamic coefficients (optional)
        timestamp: Iteration timestamp
    """
    iteration: int
    residuals: Dict[str, float] = field(default_factory=dict)
    cfl: float = 0.0
    coefficients: Optional[AerodynamicCoefficients] = None
    timestamp: Optional[datetime] = None


@dataclass
class SimulationSummary:
    """Simulation summary statistics
    
    Attributes:
        total_iterations: Total number of iterations
        final_residuals: Final residual values
        initial_coefficients: Initial aerodynamic coefficients
        final_coefficients: Final aerodynamic coefficients
        computation_time: Total computation time (seconds)
        converged: Whether simulation converged
        convergence_criteria: Convergence criteria used
    """
    total_iterations: int = 0
    final_residuals: Dict[str, float] = field(default_factory=dict)
    initial_coefficients: Optional[AerodynamicCoefficients] = None
    final_coefficients: Optional[AerodynamicCoefficients] = None
    computation_time: float = 0.0
    converged: bool = False
    convergence_criteria: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        if self.initial_coefficients:
            d['initial_coefficients'] = self.initial_coefficients.to_dict()
        else:
            d['initial_coefficients'] = None
        if self.final_coefficients:
            d['final_coefficients'] = self.final_coefficients.to_dict()
        else:
            d['final_coefficients'] = None
        return d


class ConvergenceAnalyzer:
    """Convergence history analyzer
    
    Tracks and analyzes residual history, CFL number evolution,
    and aerodynamic coefficient changes during simulation.
    
    Attributes:
        history: List of iteration data
        start_time: Simulation start time
    
    Example:
        >>> analyzer = ConvergenceAnalyzer()
        >>> analyzer.add_iteration(
        ...     iteration=1,
        ...     residuals={'continuity': 1e-2, 'momentum': 1e-3},
        ...     cfl=5.0
        ... )
        >>> analyzer.export_csv("convergence.csv")
    """
    
    def __init__(self):
        """Initialize convergence analyzer"""
        self.history: List[IterationData] = []
        self.start_time: Optional[datetime] = None
        logger.info("ConvergenceAnalyzer initialized")
    
    def add_iteration(
        self,
        iteration: int,
        residuals: Optional[Dict[str, float]] = None,
        cfl: float = 0.0,
        coefficients: Optional[AerodynamicCoefficients] = None
    ) -> None:
        """Add iteration data to history
        
        Args:
            iteration: Iteration number
            residuals: Residual values for each equation
            cfl: CFL number
            coefficients: Aerodynamic coefficients (optional)
            
        Example:
            >>> analyzer.add_iteration(
            ...     iteration=100,
            ...     residuals={'continuity': 1e-5},
            ...     cfl=10.0
            ... )
        """
        if residuals is None:
            residuals = {}
        
        data = IterationData(
            iteration=iteration,
            residuals=residuals.copy(),
            cfl=cfl,
            coefficients=coefficients,
            timestamp=datetime.now()
        )
        
        self.history.append(data)
        
        # Log progress every 10 iterations
        if iteration % 10 == 0:
            res_str = ", ".join([f"{k}={v:.2e}" for k, v in residuals.items()])
            logger.info(f"Iteration {iteration}: residuals=[{res_str}], CFL={cfl:.2f}")
    
    def export_csv(self, output_path: str) -> Path:
        """Export convergence history to CSV file
        
        Args:
            output_path: Output CSV file path
            
        Returns:
            Path: Path to exported file
            
        Raises:
            IOError: File write error
            
        Example:
            >>> path = analyzer.export_csv("convergence.csv")
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.csv')
        
        logger.info(f"Exporting convergence history to: {output_path}")
        
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                header = ['iteration', 'cfl']
                if self.history:
                    header.extend(sorted(self.history[0].residuals.keys()))
                    if self.history[0].coefficients:
                        header.extend(['Cd', 'Cl', 'Cm', 'Cs', 'Cy', 'Cr'])
                writer.writerow(header)
                
                # Write data rows
                for data in self.history:
                    row = [data.iteration, data.cfl]
                    res_keys = sorted(data.residuals.keys())
                    row.extend([data.residuals.get(k, 0.0) for k in res_keys])
                    
                    if data.coefficients:
                        coeffs = data.coefficients.to_dict()
                        row.extend([
                            coeffs.get('Cd', 0.0),
                            coeffs.get('Cl', 0.0),
                            coeffs.get('Cm', 0.0),
                            coeffs.get('Cs', 0.0),
                            coeffs.get('Cy', 0.0),
                            coeffs.get('Cr', 0.0)
                        ])
                    
                    writer.writerow(row)
            
            logger.success(f"Convergence history exported: {output_path}")
            return output_path
            
        except IOError as e:
            logger.error(f"Failed to export CSV: {e}")
            raise
    
    def get_summary(self, computation_time: float = 0.0) -> SimulationSummary:
        """Get simulation summary statistics
        
        Args:
            computation_time: Total computation time in seconds
            
        Returns:
            SimulationSummary: Summary statistics
        """
        if not self.history:
            return SimulationSummary()
        
        # Get first and last iteration
        first_iter = self.history[0]
        last_iter = self.history[-1]
        
        # Check convergence (simple criterion: residuals < 1e-4)
        converged = all(v < 1e-4 for v in last_iter.residuals.values())
        
        summary = SimulationSummary(
            total_iterations=len(self.history),
            final_residuals=last_iter.residuals.copy(),
            initial_coefficients=first_iter.coefficients,
            final_coefficients=last_iter.coefficients,
            computation_time=computation_time,
            converged=converged,
            convergence_criteria={'residual_threshold': 1e-4}
        )
        
        logger.info(f"Simulation summary:\n"
                   f"  Iterations: {summary.total_iterations}\n"
                   f"  Converged:  {summary.converged}\n"
                   f"  Time:       {summary.computation_time:.1f}s")
        
        return summary


class SimulationReport:
    """Simulation report generator
    
    Generates comprehensive JSON reports containing simulation parameters,
    convergence history, and final results.
    
    Attributes:
        config: Simulation configuration
        analyzer: Convergence analyzer
    
    Example:
        >>> report = SimulationReport(config, analyzer)
        >>> report.generate("report.json", computation_time=3600.0)
    """
    
    def __init__(
        self,
        config: dict,
        analyzer: ConvergenceAnalyzer
    ):
        """Initialize report generator
        
        Args:
            config: Simulation configuration dictionary
            analyzer: Convergence analyzer with history data
        """
        self.config = config
        self.analyzer = analyzer
        logger.info("SimulationReport initialized")
    
    def generate(
        self,
        output_path: str,
        computation_time: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> Path:
        """Generate simulation report in JSON format
        
        Args:
            output_path: Output JSON file path
            computation_time: Total computation time in seconds
            metadata: Additional metadata (optional)
            
        Returns:
            Path: Path to generated report
            
        Raises:
            IOError: File write error
            
        Example:
            >>> path = report.generate("report.json", computation_time=3600.0)
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.json')
        
        logger.info(f"Generating simulation report: {output_path}")
        
        # Build report structure
        report = {
            'metadata': {
                'software': 'AutoFlowCFD',
                'version': '0.1-MVP',
                'generated_at': datetime.now().isoformat(),
                **(metadata or {})
            },
            'configuration': self.config,
            'summary': self.analyzer.get_summary(computation_time).to_dict(),
            'convergence_history': {
                'total_iterations': len(self.analyzer.history),
                'final_residuals': self.analyzer.history[-1].residuals if self.analyzer.history else {},
            }
        }
        
        # Add aerodynamic coefficients if available
        if self.analyzer.history and self.analyzer.history[-1].coefficients:
            report['aerodynamic_coefficients'] = \
                self.analyzer.history[-1].coefficients.to_dict()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.success(f"Simulation report generated: {output_path}")
            return output_path
            
        except IOError as e:
            logger.error(f"Failed to generate report: {e}")
            raise
