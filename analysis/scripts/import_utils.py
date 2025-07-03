"""
Import utilities for evolution simulation analysis.

This module provides robust import setup that works regardless of where
the notebook/script is executed from in the project directory structure.
"""

import sys
import os

def setup_project_imports():
    """
    Automatically configure imports to work from any directory.
    
    Searches for the 'src' directory in common locations and adds the
    project root to Python path. This allows notebooks to be run from
    anywhere in the project structure.
    
    Returns:
        str: Path to the project root directory
        
    Raises:
        ImportError: If src/ directory cannot be located
    """
    current_dir = os.getcwd()
    
    # Common scenarios where code might be run from:
    search_paths = [
        current_dir,                                    # Run from project root
        os.path.dirname(current_dir),                   # Run from analysis/ subdirectory  
        os.path.dirname(os.path.dirname(current_dir)),  # Run from deeper subdirectory
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),  # Run from analysis/scripts/
    ]
    
    for search_dir in search_paths:
        src_path = os.path.join(search_dir, 'src')
        if os.path.exists(src_path) and os.path.isdir(src_path):
            # Found src directory, add project root to path
            if search_dir not in sys.path:
                sys.path.insert(0, search_dir)
            
            print(f"✅ Project root: {search_dir}")
            print(f"✅ Found src/ at: {src_path}")
            return search_dir
    
    # Debug information if src not found
    print(f"❌ Could not find src/ directory")
    print(f"Current directory: {current_dir}")
    print(f"Directory contents: {os.listdir(current_dir)}")
    print(f"Searched in: {search_paths}")
    
    raise ImportError(
        "Cannot locate src/ directory. Please run from project root or analysis/ subdirectory."
    )

def import_simulation_modules():
    """
    Import simulation modules with automatic path setup.
    
    Returns:
        tuple: (Simulation class, project_root_path)
    """
    project_root = setup_project_imports()
    
    try:
        from src.simulation import Simulation
        print("✅ Successfully imported simulation modules")
        return Simulation, project_root
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure all source files are in src/ directory")
        raise

# Convenience function for notebooks
def quick_setup():
    """
    One-line setup for notebooks.
    
    Returns:
        Simulation: The Simulation class, ready to use
    """
    Simulation, _ = import_simulation_modules()
    return Simulation 