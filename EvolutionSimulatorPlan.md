# Project Plan: A Validated Simulation of Evolution

**Document Version:** 1.0
**Last Updated:** July 2, 2025

## 0. Note on using this document
This document is intended primarily for a coding agent to use to understand the principles and goals behind the project but also to note and maintain any design decisions and implementation characteristics necessary for further clean and robust development.

As such, it is a living document and should also be updated as the project progresses. The document is not meant to constraint any developers but to give context and structure, background to help what's gone before. 

In particular, the July 2nd version (please update the updated date if you change the document) is quite preliminary, as it waw written before ay development, so it's very likely that details will change as we start to create a project.

## 1. Project Vision & Core Principles

### 1.1. Primary Goal
To develop a flexible and reliable simulation environment to model and compare the evolutionary dynamics of **sexual vs. asexual reproduction**. The simulation will serve as a tool to explore how different reproductive strategies affect adaptation, genetic diversity, and population fitness under various selective pressures.

### 1.2. Core Principles
This project is guided by a commitment to scientific and software engineering rigor.
* **Trustworthiness First:** The primary measure of success is the simulation's reliability. Every step will be taken to verify the code's correctness and validate its output against established scientific models.
* **Incremental Complexity:** We will begin with the simplest possible models and add complexity only after the existing foundation is tested and validated.
* **Clarity and Control:** We will build the core simulation logic from scratch to ensure a complete understanding and control over every component, avoiding "black box" implementations for the foundational engine.

## 2. Technical Framework

### 2.1. Core Technology Stack
* **Language:** Python 3.x
* **Core Libraries:**
  * **NumPy:** For efficient, high-performance representation and manipulation of populations and genomes.
  * **Matplotlib / Seaborn:** For generating plots and visualizations for analysis and validation.
  * **Pandas:** For handling and analyzing simulation output data.
  * **SciPy:** For statistical tests required during the validation phase.
* **Testing Framework:** `pytest` (or Python's built-in `unittest`).

### 2.2. Development Methodology
* **Test-Driven Development (TDD):** We will use a TDD approach for developing the core simulation components. For each component, we will:
  1. Write a test that defines the desired functionality.
  2. Confirm the test fails.
  3. Write the simplest implementation code to make the test pass.
  4. Refactor as needed.
* **Version Control:** (To be decided) A Git repository is recommended for tracking changes.

### 2.3. Cross-Validation Strategy
* **External Tool:** **DEAP** (Distributed Evolutionary Algorithms in Python).
* **Purpose:** After our own simulation engine is built and passes its own validation tests, we will use DEAP as an independent, external benchmark. We will configure DEAP to run an identical simulation and compare the statistical outputs to cross-validate our implementation.

### 2.4. Parameterization and Configuration
All key simulation parameters (e.g., population size, genome length, mutation rate, selection method) will be configurable via a central configuration file or command-line arguments. This will allow for easy experimentation and reproducibility.

### 2.5. Random Seed Control
To ensure reproducibility, each simulation run will allow explicit control of the random seed. The seed value will be logged alongside simulation outputs.

### 2.6. Documentation Approach
The codebase will use Python docstrings for all public classes and functions, with additional documentation for complex or user-facing components. As the project grows, we may adopt a documentation generator such as Sphinx or MkDocs.

## 3. High-Level Implementation Plan

The project will proceed in distinct, sequential phases.

### Phase 1: Foundational Implementation & Verification (Sexual Reproduction Model)
* **Objective:** Build and verify the core mechanics of a sexual reproduction model.
* **Methodology:** Strict TDD.
* **Key Tasks:**
  1. Set up the project structure, including test directories.
  2. Implement a `Genome` representation (initial plan: 1D NumPy binary array).
  3. Implement a `Population` container (initial plan: 2D NumPy array).
  4. Implement and test core genetic operators:
     * `selection()`: (Decision needed: Tournament selection is a good starting point).
     * `crossover()`: (Initial plan: Single-point crossover).
     * `mutation()`: (Initial plan: Per-bit mutation with a fixed rate).
  5. Assemble the operators into a main `simulation_loop()`.

### Phase 2: Scientific Validation
* **Objective:** Validate the simulation's behavior against established principles of population genetics.
* **Key Tasks:**
  1. **Neutral Drift Validation:**
     * Implement a "no selection" mode where parents are chosen randomly.
     * Write and run tests to confirm that the fixation probability of a neutral allele matches its initial frequency ($p$).
     * Write and run tests to confirm that heterozygosity decays at the theoretical rate of $1/(2N)$.
  2. **Simple Selection Validation:**
     * Implement a simple, direct fitness function (e.g., additive fitness based on the count of '1's).
     * Write and run tests to confirm that the fixation probability of a single beneficial mutation approximates the theoretical value of $2s$.
  3. **Iterative Validation**
     * Continue to generate validation procedures and check for known results and available software to validate our simulations.

### Phase 3: Asexual Model & Comparative Analysis
* **Objective:** Implement the asexual model and run the core comparative experiments.
* **Key Tasks:**
  1. Implement an asexual reproduction operator (cloning with mutation).
  2. Design experiments to compare sexual vs. asexual populations. Key metrics to track:
     * Rate of adaptation (speed of mean fitness increase).
     * Maintenance of genetic diversity.
     * Resilience to environmental change (requires implementing a dynamic fitness landscape).
  3. Run experiments and analyze results.

### Phase 4: Expansion & Frontend (Future Work)
* **Objective:** Extend the model and create an interactive interface.
* **Key Tasks:**
  * Explore more complex genetic models (e.g., continuous-valued genomes, epistasis).
  * Develop a web-based frontend using a Python web framework (e.g., Flask, FastAPI) and a JavaScript visualization library.

## 4. Open Questions & Initial Parameters

This section will be updated as we make decisions.

* **Initial Population Size (**$N$**):** TBD (e.g., 100)
* **Initial Genome Length (**$L$**):** TBD (e.g., 50)
* **Initial Mutation Rate (**$\mu$**):** TBD (e.g., 0.001 per bit)
* **Selection Mechanism for Phase 1:** TBD (Tournament selection is proposed).
* **Fitness Function for Phase 2:** TBD (Simple additive model, `fitness = sum(genome)`, is proposed).
* **Data Logging Format:** TBD (CSV is a simple starting point).