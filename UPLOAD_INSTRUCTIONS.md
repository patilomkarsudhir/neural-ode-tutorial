# Upload Instructions

Due to file size limitations of the GitHub API, some files need to be uploaded manually:

## Files to Upload Manually

1. **neural_ode_tutorial.tex** (49 KB) - LaTeX source file
2. **neural_ode_tutorial.pdf** (591 KB) - Compiled PDF with figures  
3. **neural_ode_demo.ipynb** - Jupyter notebook with experiments
4. **saved_models/neural_ode_residual64_optimized.pth** - Trained model weights
5. **figures/** directory - All 9 publication-quality figures (PDF and PNG)

## Upload Command

From the local directory, run:

`bash
git clone https://github.com/patilomkarsudhir/neural-ode-tutorial.git
cd neural-ode-tutorial
cp ../path/to/files/* .
git add .
git commit -m \"Add LaTeX tutorial, notebook, figures, and trained models\"
git push origin main
`

Or use GitHub's web interface to upload these files directly.

## Alternative: Git LFS

For large files like the PDF and model weights, consider using Git LFS:

`bash
git lfs track \"*.pdf\"
git lfs track \"*.pth\"
git lfs track \"figures/*.pdf\"
git lfs track \"figures/*.png\"
`
