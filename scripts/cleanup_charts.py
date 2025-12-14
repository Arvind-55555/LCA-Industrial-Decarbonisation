#!/usr/bin/env python3
"""
Clean up charts - keep only specified ones
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path

# Charts to keep
KEEP_CHARTS = [
    "cement_lca_breakdown.png",
    "sector_comparison.png",
    "steel_lca_breakdown.png"
]

def cleanup_charts(plots_dir: str = "outputs/plots"):
    """Remove charts not in the keep list"""
    plots_path = Path(plots_dir)
    
    if not plots_path.exists():
        print(f"Plots directory not found: {plots_dir}")
        return
    
    all_files = list(plots_path.glob("*.png"))
    all_files.extend(plots_path.glob("*.jpg"))
    all_files.extend(plots_path.glob("*.jpeg"))
    
    removed = 0
    kept = 0
    
    for file in all_files:
        if file.name in KEEP_CHARTS:
            kept += 1
            print(f"✅ Keeping: {file.name}")
        else:
            file.unlink()
            removed += 1
            print(f"🗑️  Removed: {file.name}")
    
    print(f"\nSummary: Kept {kept} charts, Removed {removed} charts")

if __name__ == "__main__":
    cleanup_charts()

