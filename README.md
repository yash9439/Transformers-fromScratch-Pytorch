# ANLP Assignment 3 Report

Link to Saved Model : https://iiitaphyd-my.sharepoint.com/:u:/g/personal/yash_bhaskar_research_iiit_ac_in/ES4Fa8vVpy5LuNI6JnVTfOkBDb5eTQ51Nje-MdUz74aaIQ?e=ZVx6Oa

This repository contains a report for the Assignment 3 of the Advanced Natural Language Processing (ANLP) course. The report discusses the key concepts of transformer architecture, self-attention, positional encodings, and hyperparameter tuning for machine translation tasks.

## Report Structure

The report is organized into the following sections:

1. **Abstract:** An overview of the key components of the transformer architecture and their significance in natural language processing.

2. **Theory Question 1:** Explanation of the purpose of self-attention and how it facilitates capturing dependencies in sequences.

3. **Theory Question 2:** Detailed information on why transformers use positional encodings and how they are incorporated into the transformer architecture.

4. **Hyperparameter Tuning:** An analysis of the model's performance across different hyperparameter configurations, including BLEU scores and loss curves.

5. **Analysis:** Discussion of the results, significance of hyperparameters, and a summary of the best configuration.

## Hyperparameter Configurations

The report includes an analysis of the model's performance across various hyperparameter configurations. Three sets of hyperparameters are considered, differing in the number of attention heads (h). Each configuration is evaluated based on BLEU scores and training loss. The best configuration is selected based on these metrics.

## Best Configuration

The best hyperparameter configuration is chosen based on the performance evaluation, and it is as follows:

- Learning Rate (lr): 0.01
- Batch Size: 32
- Dropout: 0.3
- Number of Attention Heads (h): 16
- Number of Layers (N): 2
- Number of Epochs: 10
- Model Dimension (d_model): 512
- Feed-Forward Dimension (d_ff): 2048

## Test Data Score

The model's performance on test data is evaluated using the BLEU score, and the obtained score is 1.709664057978683.

## Report Author

- **Author:** Yash Bhaskar
- **Roll Number:** 2021114012
