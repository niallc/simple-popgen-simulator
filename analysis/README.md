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