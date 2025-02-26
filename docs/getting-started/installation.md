# Installation

There are several ways to install the Narrative LLM Tools package.

## Using pip

The simplest way to install the package is using pip:

```bash
pip install narrative-llm-tools
```

## From Source

For the latest development version, you can install directly from GitHub:

```bash
pip install git+https://github.com/narrative-io/narrative-llm-tools.git
```

## For Development

If you want to contribute to the package:

```bash
# Clone the repository
git clone https://github.com/narrative-io/narrative-llm-tools.git
cd narrative-llm-tools

# Install in development mode
pip install -e .
```

## In a HuggingFace Inference Endpoint

To use the HuggingFace Inference Endpoints Handler, include this in your `requirements.txt`:

```text
narrative-llm-tools @ git+https://github.com/narrative-io/narrative-llm-tools.git@v0.1.5
```

Additionally, create a `handler.py` file in your repository:

```python
from narrative_llm_tools.handlers.huggingface import EndpointHandler

handler = EndpointHandler()
```

## Dependencies

The package requires:

- Python 3.10+
- Key dependencies:
  - `pydantic`: For data validation and settings management
  - `requests`: For HTTP requests
  - `cachetools`: For performance optimizations
  - And other dependencies as listed in `pyproject.toml`

## Verifying Installation

After installation, you can verify everything is working correctly:

```python
import narrative_llm_tools

print(f"Successfully installed narrative-llm-tools version {narrative_llm_tools.__version__}")
```
