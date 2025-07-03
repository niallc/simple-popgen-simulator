# Evolution Simulation Analysis

This directory contains comprehensive analysis tools for the evolution simulation project.

## Main Analysis Notebook: `corrected_simulation_analysis.ipynb`

### Features

1. **Comprehensive Scenario Comparison**
   - 6 scenarios: 3 selection regimes × 2 reproduction modes
   - True neutral evolution vs. additive selection vs. weak selection
   - Asexual vs. sexual reproduction

2. **Theoretical Validation**
   - Population genetics predictions for heterozygosity decay
   - Correlation analysis between observed and theoretical values
   - Validates simulation accuracy against known theory

3. **Rich Visualizations**
   - Fitness evolution comparison plots
   - Heterozygosity analysis with theoretical overlays
   - Side-by-side reproduction mode comparisons

4. **Comprehensive Summary Tables**
   - Detailed metrics for all scenarios
   - Fitness changes, heterozygosity retention rates
   - Easy-to-copy text summaries for further analysis

### Key Insights Captured

- **Selection vs. Neutral**: Proper distinction between true neutral evolution (equal fitness) and additive selection
- **Theoretical Fit**: How well simulations match population genetics theory
- **Reproduction Mode Effects**: Sexual vs. asexual impact on genetic diversity
- **Selection Strength**: Comparison of different selection intensities

### Usage

```python
# Run the notebook to get comprehensive analysis including:
# - All 6 scenario simulations
# - Theoretical predictions vs. observations
# - Summary statistics and visualizations
# - Text-based summaries ready for LLM analysis
```

### Output Summary Format

The notebook generates detailed text summaries like:

```
COMPREHENSIVE EVOLUTION SIMULATION SUMMARY
==========================================

Scenario             Init Fit Final Fit Fit Change Fit Change %
Init Het Final Het Het Retention Het Decay Rate
------------------------------------
Asexual Neutral      10.22    10.22    +0.00      +0.0%
0.506    0.506    100.0%       0.0000

[... detailed table for all scenarios ...]

KEY EVOLUTIONARY PATTERNS
==========================
1. FITNESS EVOLUTION:
   • Selection vs. Neutral: ✓ (patterns match expectations)
2. HETEROZYGOSITY RETENTION:
   • Sexual maintains more diversity: ✓
3. THEORETICAL PREDICTIONS:
   • Simulations match theory: ✓
```

This format is ideal for:
- Quick pattern recognition
- Sharing with collaborators
- Input to LLMs for further analysis
- Validation of simulation behavior

## Import Solutions

### The Problem
Jupyter notebooks have import issues when run from different directories. Running from project root vs. analysis/ subdirectory causes different import behaviors.

### The Solution
We provide multiple robust import patterns:

#### Option 1: Use Simple Hardcoded Import (Recommended)
```python
import sys
import os

# Simple, hardcoded project root
project_root = os.path.expanduser("~/Documents/programming/SelectionAndSexualReproduction")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation import Simulation
```

#### Option 2: Legacy Try-Catch Pattern (No longer needed)
```python
# This complex pattern is no longer needed with the simple hardcoded approach
try:
    from src.simulation import Simulation
except ImportError:
    import sys, os
    if os.path.exists('src'):
        sys.path.insert(0, os.getcwd())
    elif os.path.exists('../src'):
        sys.path.insert(0, os.path.dirname(os.getcwd()))
    from src.simulation import Simulation
```

#### Option 3: Run from Specific Directory
Always `cd` to project root before running notebooks.

### Files
- `notebook_template.ipynb` - Template showing best practices
- `corrected_simulation_analysis.ipynb` - Full analysis with simple imports

### Requirements

- Works from project root or analysis/ subdirectory
- Compatible with pytest test suite
- Graceful fallback between import methods 