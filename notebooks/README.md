# Notebook Workflow Guide

## Local Development (Cursor + Jupyter)
- Use these notebooks for prototyping and development
- Run on local machine with smaller datasets
- Focus on code development and debugging

## Production Training (Google Colab)
- Use `_colab.ipynb` versions for GPU/TPU intensive tasks
- Sync with local versions before major training runs
- Copy final models back to local for integration

## Workflow Steps:
1. **Develop locally**: Create/edit notebooks in Cursor
2. **Test locally**: Run with sample data using local Jupyter
3. **Sync to Colab**: Upload notebook to Colab for GPU training
4. **Train in Colab**: Run intensive training with full datasets
5. **Sync results back**: Download trained models and results
6. **Integrate locally**: Use trained models in your pipeline

## File Naming Convention:
- `{name}.ipynb` - Local development version
- `{name}_colab.ipynb` - Colab production version
- `{name}_experiments.ipynb` - Quick experiments and tests 