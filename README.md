# Narrative LLM Tools

A set of tools to use when calling Narrative I/O LLMs. This package provides utilities for managing LLM conversations, handling REST API interactions, and enforcing JSON schema formats.

## Features

- 🤗 HuggingFace Inference Endpoints Handler
- 🌐 RestAPI Client
- 💬 Conversation State Management
- ✅ JSON Schema Validation and Enforcement
- 🛠️ Tool Management System
- 💾 Caching System for Format Enforcers

## HuggingFace Space

Check out our [HuggingFace Space](https://huggingface.co/spaces/narrative-io/README) for interactive demos and more information about using Narrative I/O LLMs.

The HuggingFace Space repository is included as a git submodule in this project under the `hf-space/` directory. When cloning this repository, use `git clone --recurse-submodules` to also get the Space files.

## Installation

You can install the package using pip:

```bash
pip install narrative-llm-tools
```

## Usage

### HuggingFace Inference Endpoints Handler

To use the HuggingFace Inference Endpoints Handler, via HuggingFace Inference Endpoints you will need to reference this library from a `requirements.txt` file in your repository:

```text
narrative-llm-tools @ git+https://github.com/narrative-io/narrative-llm-tools.git@v0.1.5
```

Additionally, you will need to add the following `handler.py` file to your repository:

```python
from narrative_llm_tools.handlers.huggingface import EndpointHandler
```

## License

MIT License
