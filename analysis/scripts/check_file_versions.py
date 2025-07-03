#!/usr/bin/env python3
"""
File Version Checker Utility

This script helps you check existing file versions and see what the next version would be.
Useful for understanding the versioning system before running analysis scripts.
"""

import sys
import os

# Simple, hardcoded project root - clean and obvious
project_root = os.path.expanduser("~/Documents/programming/SelectionAndSexualReproduction")

# Add project root to Python path for imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sexual_vs_asexual_analysis import list_file_versions, generate_filename, SexualVsAsexualAnalyzer

# Set up default data directory
default_data_dir = os.path.join(project_root, 'analysis', 'data', 'sexual_vs_asexual')

def main():
    """Main function to check file versions."""
    
    if len(sys.argv) > 1:
        # Check specific file
        file_path = sys.argv[1]
        list_file_versions(file_path)
    else:
        # Show example with default parameters
        print("📁 File Version Checker")
        print("=" * 50)
        print("Usage: python check_file_versions.py [file_path]")
        print()
        
        # Show example with default parameters
        analyzer = SexualVsAsexualAnalyzer()
        example_filename = generate_filename(analyzer, "comprehensive")
        example_path = os.path.join(default_data_dir, example_filename)
        
        print("Example with default parameters:")
        list_file_versions(example_path)
        
        print("\n" + "=" * 50)
        print("To check a specific file:")
        print("  python check_file_versions.py path/to/your/file.csv")
        print()
        print("To check the default analysis file:")
        print(f"  python check_file_versions.py {os.path.join(default_data_dir, 'sexual_vs_asexual_results.csv')}")

if __name__ == "__main__":
    main() 