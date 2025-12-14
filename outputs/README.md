# Outputs Directory

This directory contains all generated visualizations, reports, and results.

## 📁 Directory Structure

```
outputs/
├── plots/              # Generated chart images (PNG)
│   ├── steel_lca_breakdown.png
│   ├── cement_lca_breakdown.png
│   ├── policy_impact_comparison.png
│   ├── sector_comparison.png
│   └── grid_ci_*.png
├── reports/            # HTML summary reports
│   └── visualization_summary.html
└── README.md          # This file
```

## 📊 Generated Visualizations

### LCA Breakdown Charts
- `steel_lca_breakdown.png` - Steel H2-DRI emission breakdown
- `cement_lca_breakdown.png` - Cement CCUS emission breakdown

### Policy Impact Charts
- `policy_impact_comparison.png` - Comprehensive policy comparison
- `policy_impact.png` - Policy impact visualization

### Sector Analysis
- `sector_comparison.png` - Cross-sector emission comparison

### Grid Carbon Intensity
- `grid_ci_timeseries.png` - Time series analysis
- `grid_ci_us.png` - US grid data
- `grid_ci_eu.png` - EU grid data
- `grid_ci_de.png` - Germany grid data
- `grid_ci_fr.png` - France grid data

## 📄 Reports

### HTML Summary Report
- **File**: `reports/visualization_summary.html`
- **Contains**: All charts, results tables, metrics
- **View**: Open in any web browser

## 🚀 Generating Outputs

### Generate All Visualizations
```bash
python scripts/generate_visualization_summary.py
```

### View Interactive Dashboard
```bash
python run_dashboard.py results
```
Then visit: http://localhost:8050

### View HTML Report
```bash
# Open in browser
xdg-open outputs/reports/visualization_summary.html
```

## 📈 Results Summary

See [docs/VISUALIZATION_INDEX.md](../docs/VISUALIZATION_INDEX.md) for complete results summary.

## 🖼️ Image Gallery

All images are high-resolution PNG files (300 DPI) suitable for:
- Presentations
- Reports
- Publications
- Documentation

## 📝 Notes

- Images are regenerated when running visualization scripts
- Old images are overwritten
- HTML reports include embedded images
- Dashboard displays images dynamically

