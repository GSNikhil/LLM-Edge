# LLM-Edge
## Installation
1. **Install Required Libraries**  
Run the following command to install necessary libraries:  
```bash
pip install transformers datasets torch rouge_score
```

## Usage Overview
2. The `main.ipynb` notebook provides a step-by-step guide to:
- Install all required libraries
- Download models and datasets
- Establish baseline performance metrics
- Evaluate a fine-tuned model
- Perform Activation-aware Weight Quantization (AWQ)
- Prune the model using Norm and Sensitivity-based methods
- Quantize the best-performing pruned model

## File Descriptions
3. **`awq_quantizer.py`** - Contains functions and utilities for AWQ.  
4. **`download_datasets_models.py`** - Includes functions to download and save datasets and models.  
5. **`evaluate_llm.py`** - Provides functions to calculate model performance metrics.  
6. **`prune_utils.py`** - Implements Norm and Sensitivity-based model pruning techniques.  
7. **`run_qemu.py`** - Contains functions for measuring model throughput using the QEMU simulator.  
8. **`util_functions.py`** - Provides utilities to retrieve model size and parameter information.