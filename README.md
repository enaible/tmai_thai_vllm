# vLLM Solar MoE Plugin

A vLLM out-of-tree plugin for serving Solar MoE models with high-performance inference capabilities.

## Overview

This plugin enables vLLM to serve Solar MoE (Mixture of Experts) models efficiently, providing:

- **High-performance inference** with vLLM's optimized engine
- **MoE support** for Solar MoE architectures
- **Easy integration** with existing vLLM workflows
- **Memory optimization** for large MoE models

## Features

- ✅ Full Solar MoE model support
- ✅ vLLM compatibility with proper tensor handling
- ✅ Optimized memory usage for MoE layers
- ✅ Support for custom model configurations
- ✅ Easy plugin installation and setup

## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.9 or higher
- CUDA-compatible GPU
- pip package manager

### Step 1: Install Core Dependencies

Install the required dependencies for vLLM and flash attention:

```bash
pip install vllm flash-attn flashinfer-python accelerate
```

### Step 2: Install the Plugin

Install the Solar MoE plugin:

```bash
pip install -e .
```

This will install the `solar-moe-vllm-plugin` package and register it with vLLM.

### Step 3: Set Environment Variable

Set the vLLM plugins environment variable:

```bash
export VLLM_PLUGINS=solar_moe_plugin
```

### Step 4: Serve Your Model

Start serving your Solar MoE model:

```bash
vllm serve vessl/thai-tmai --trust-remote-code --no-enable-prefix-caching
```

Optional for NAS
```bash
vllm serve vessl/thai-tmai --trust-remote-code --no-enable-prefix-caching --download-dir ./model_cache
```

Local Directory
```bash
vllm serve /root/nas/updated_final_model --trust-remote-code --no-enable-prefix-caching --enforce_eager --no-enable_chunked_prefill --convert="none"
```
