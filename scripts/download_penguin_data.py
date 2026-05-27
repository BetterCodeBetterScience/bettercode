#!/usr/bin/env python3
"""
Download penguin data files to be included as package data.

This script downloads the Palmer Penguins data from the original source
and saves it to the package data directory. This documents the data provenance
and allows the data to be shipped with the package.
"""

import sys
from pathlib import Path

# Add src to path to import bettercode
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bettercode.penguindata import DATA_URLS, load_penguin_data


def download_all_species():
    """Download all penguin species data files."""
    print("Downloading Palmer Penguins data from original sources...\n")
    
    for species in ['adelie', 'gentoo', 'chinstrap']:
        print(f"Downloading {species.title()} penguin data...")
        load_penguin_data(species, force_download=True)
        print()
    
    print("All data files downloaded successfully!")
    print("\nData sources:")
    for species, url in DATA_URLS.items():
        print(f"  {species.title()}: {url}")


if __name__ == '__main__':
    download_all_species()
