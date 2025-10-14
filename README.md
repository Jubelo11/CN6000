# Week 2 — Gradient Descent & Multiple Linear Regression (CN6000)

**Author:** Jubelo Oladimeji  
**Module:** CN6000

## What this repo shows
- **Gradient Descent (from scratch):** A tiny univariate example to show parameter updates and a convergence plot.
- **Multiple Linear Regression (scikit-learn):** Trained on the California Housing dataset with evaluation metrics (MSE, RMSE, MAE, R²). Also compares using **all features** vs **high-correlation features**.

## Why it’s useful
- Demonstrates the core idea behind Linear Regression and how Gradient Descent finds a good line.
- Shows a practical model using real data and standard evaluation metrics.
- Clean, readable code suitable for coursework, interviews, or portfolio.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install numpy scikit-learn matplotlib pandas
```

## Run
```bash
python Week2_GradientDescent_and_LinearRegression.py
```

You’ll see:
- A scatter + fitted-line plot for the toy dataset
- A convergence plot (cost vs. iteration) for Gradient Descent
- Printed metrics for the California Housing regression task

## Notes
- The correlation threshold for feature selection is **0.5** by default. If you want more features, change it to **0.15** in the `demo_linear_regression` call.
- This code avoids unnecessary complexity to keep it easy to read and extend.
- Plots use plain matplotlib so they work in most environments.

---

*Feel free to fork or adapt for your own learning and portfolio.*
