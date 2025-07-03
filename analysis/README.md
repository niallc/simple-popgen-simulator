# Analysis Directory

This directory contains all analysis-related files for the evolution simulator project, organized into a clear structure for better file management and reproducibility.

## Directory Structure

```
analysis/
├── data/                          # Data files and outputs
│   ├── sexual_vs_asexual/         # Sexual vs asexual reproduction results
│   └── archive/                   # Archived data files
├── notebooks/                     # Jupyter notebooks for analysis
├── scripts/                       # Python scripts for data generation and analysis
├── results/                       # Documentation and analysis results
└── README.md                      # This file
```

## File Organization

### Data Files (`data/`)
- **`sexual_vs_asexual/`**: Contains CSV files with simulation results comparing sexual vs asexual reproduction
- **`archive/`**: For older data files that are no longer actively used

### Scripts (`scripts/`)
- **`sexual_vs_asexual_analysis.py`**: Main script for running comprehensive sexual vs asexual reproduction comparisons
- **`import_utils.py`**: Utilities for robust import handling across different directory structures
- **`quick_test.py`**: Quick testing and validation scripts
- **`check_file_versions.py`**: Utility to check existing file versions and see what the next version would be

### Notebooks (`notebooks/`)
- Jupyter notebooks for interactive analysis and visualization
- Each notebook focuses on specific aspects of the simulation results

### Results (`results/`)
- **`CODEBASE_ANALYSIS.md`**: Analysis of the codebase structure and usage
- **`EvolutionSimulatorPlan.md`**: Original project planning document
- **`glossary.md`**: Terminology and definitions
- **`README.md`**: Analysis-specific documentation

## Usage

### Running Analysis Scripts
From the `analysis/` directory:
```bash
# Run the main sexual vs asexual analysis
python scripts/sexual_vs_asexual_analysis.py

# Run quick tests
python scripts/quick_test.py
```

### Working with Notebooks
From the `analysis/` directory:
```bash
# Start Jupyter notebook server
jupyter notebook notebooks/
```

### Data File Naming Convention
When creating new data files, use descriptive names that include:
- Key parameters (e.g., `pop100_gen20_mut0.005`)
- Timestamp (e.g., `2024-01-15_14-30`)
- Description (e.g., `sexual_vs_asexual_comparison`)

Example: `sexual_vs_asexual_weak_selection_pop100_gen20_mut0.005_gens100_runs100_2024-01-15_14-30.csv`

**Current Implementation:**
The `sexual_vs_asexual_analysis.py` script automatically generates descriptive filenames using the format:
`sexual_vs_asexual_{fitness_regime}_{pop_size}_gen{genome_length}_mut{mutation_rate}_gens{generations}_runs{n_runs}_{timestamp}.csv`

This prevents file clobbering and makes it easy to identify the parameters used for each simulation run.

## File Versioning System

The analysis scripts include an **anti-clobbering system** that automatically prevents overwriting existing files:

### How It Works
1. **Base File Check**: If the intended filename doesn't exist, it's used as-is
2. **Version Detection**: If the file exists, the system looks for existing versions (`_v2`, `_v3`, etc.)
3. **Next Version**: Automatically creates the next available version number
4. **Fallback**: If directory access fails, adds a timestamp suffix

### Examples
- `results.csv` (doesn't exist) → saves as `results.csv`
- `results.csv` (exists) → saves as `results_v2.csv`
- `results.csv` and `results_v2.csv` exist → saves as `results_v3.csv`

### Version Checking
Use the utility script to check existing versions:
```bash
# Check a specific file
python scripts/check_file_versions.py data/your_file.csv

# Check the default analysis file
python scripts/check_file_versions.py ../data/sexual_vs_asexual/sexual_vs_asexual_results.csv
```

### Benefits
- **No data loss**: Never accidentally overwrite important results
- **Clear history**: Easy to see the progression of experiments
- **Automatic**: No manual intervention required
- **Robust**: Handles edge cases and permission errors gracefully

## Import Handling

The scripts use robust import utilities (`import_utils.py`) that automatically detect the project structure and set up imports correctly, regardless of where the script is run from.

## Version Control

- Data files are typically excluded from version control (see `.gitignore`)
- Scripts and notebooks are version controlled
- Results documentation is version controlled
- Consider using Git LFS for large data files if needed

## Best Practices

1. **Reproducibility**: Always use fixed random seeds for reproducible results
2. **Documentation**: Document parameters and assumptions in scripts and notebooks
3. **Data Validation**: Include sanity checks and validation in analysis scripts
4. **Backup**: Archive important results in the `data/archive/` directory
5. **Naming**: Use clear, descriptive names for all files 