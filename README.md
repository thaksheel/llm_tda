# LLM TDA Methods

This repository is a work in progress focused on building a pipeline for training data attribution (TDA) in large language model (LLM) settings.

The goal is to implement and evaluate the latest training data analysis (TDA) attributing model behavior back to training data. The project is centered on understanding how examples in the training set affect model outputs, metrics, and downstream performance across attribution and influence analysis. 

## Current focus

- Collecting latest TDA methods for LLMs
- Evaluating performance on influence (tail-patch score, and LDS) and attribution (MRR, P@K, R@K) metrics
- Integration of latest TDA datasets such as TRex and FTRACE-TRex 

## Repository overview

This codebase contains scripts and modules for:

- fine-tuning and experiment orchestration
- TDA / influence-based tracing workflows
- dataset handling and evaluation utilities
- experimental runs for attribution-focused analysis
