---
title: WeightSpec
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: true
license: apache-2.0
short_description: Verify LLM weight edits bit for bit; see what weights write
tags:
  - weight-space-programming
  - model-editing
  - interpretability
---

# WeightSpec Space

Verify [WeightSpec](https://github.com/namashworks/weightspec) patches against public base weights, inspect what a weight column writes into the residual stream, and rebuild the 1,023-value Qwen3.5 MATH edit from Hu et al. (2026) live.

Deploy: create a Gradio Space and upload `app.py`, `requirements.txt` and this `README.md`.
