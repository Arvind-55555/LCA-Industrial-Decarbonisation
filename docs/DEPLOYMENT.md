# GitHub Pages Deployment Guide

## 🚀 Deploy Dashboard to GitHub Pages

The dashboard is configured to deploy automatically to GitHub Pages at:
**https://arvind-55555.github.io/LCA-Industrial-Decarbonisation**

## 📋 Prerequisites

1. **GitHub Repository**: Ensure your repository is on GitHub
2. **GitHub Pages Enabled**: 
   - Go to repository Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` or `master`, folder: `/docs`

## 🔄 Automatic Deployment

The project includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that automatically:

1. ✅ Builds the static dashboard when you push to `main`/`master`
2. ✅ Generates HTML from dashboard data
3. ✅ Deploys to GitHub Pages

### Manual Trigger

You can also trigger deployment manually:
- Go to Actions tab in GitHub
- Select "Deploy to GitHub Pages" workflow
- Click "Run workflow"

## 📝 Manual Deployment Steps

If you prefer to deploy manually:

### 1. Generate Static Dashboard

```bash
python scripts/generate_static_dashboard.py
```

This creates `docs/index.html` with:
- Interactive Plotly.js charts
- All generated visualization images
- Tabbed interface (LCA Results, Grid CI, Sector Comparison, Charts)

### 2. Commit and Push

```bash
git add docs/
git commit -m "Deploy dashboard to GitHub Pages"
git push origin main
```

### 3. Enable GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Under **Source**, select:
   - Branch: `main` (or `master`)
   - Folder: `/docs`
3. Click **Save**

### 4. Access Dashboard

After a few minutes, your dashboard will be available at:
**https://arvind-55555.github.io/LCA-Industrial-Decarbonisation**

## 📊 Dashboard Features

The static dashboard includes:

- **📊 LCA Results**: Baseline vs optimized emissions with improvement opportunities
- **🌍 Grid Carbon Intensity**: Interactive time-series with time period selection
- **🏭 Sector Comparison**: Cross-sector optimization potential
- **🖼️ Charts & Images**: Gallery of generated LCA breakdown charts

## 🔧 Customization

### Update Dashboard Data

Edit `scripts/generate_static_dashboard.py` to:
- Change sample data
- Add new charts
- Modify styling
- Update layout

### Add New Charts

1. Generate charts using the workflow:
   ```bash
   python scripts/run_complete_ml_workflow.py
   ```

2. Charts will be automatically included in the static dashboard if they're in `outputs/plots/` and match the allowed list.

## 🐛 Troubleshooting

### Dashboard Not Updating

1. Check GitHub Actions workflow status
2. Verify `docs/index.html` exists
3. Ensure GitHub Pages is enabled in repository settings
4. Wait 1-2 minutes for deployment to complete

### Charts Not Showing

1. Ensure charts are generated: `python scripts/run_complete_ml_workflow.py`
2. Check that charts are in `outputs/plots/`
3. Verify chart names match allowed list in `generate_static_dashboard.py`

### Build Failures

1. Check GitHub Actions logs
2. Verify Python dependencies are installed
3. Ensure all required files exist

## 📚 Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Plotly.js Documentation](https://plotly.com/javascript/)

## ✅ Deployment Checklist

- [ ] Repository is on GitHub
- [ ] GitHub Pages is enabled (Settings → Pages)
- [ ] Workflow file exists (`.github/workflows/deploy.yml`)
- [ ] `docs/index.html` is generated
- [ ] Charts are in `outputs/plots/`
- [ ] Changes are committed and pushed
- [ ] GitHub Actions workflow completed successfully

## 🎉 Success!

Once deployed, your dashboard will be live at:
**https://arvind-55555.github.io/LCA-Industrial-Decarbonisation**

The dashboard updates automatically whenever you push changes to the `main` branch!
