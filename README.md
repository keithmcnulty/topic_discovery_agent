# document_categorization_agent

Agent to categorize documents into topics using Anthropic LLMs.

See [test_notebook.ipynb](test_notebook.ipynb) for usage.

To install the required dependencies:

Using `poetry`:
* Ensure `poetry` is installed.  If not, install it in the terminal using `brew install poetry`.
* Navigate to the root directory of this repository.
* Run `poetry install` to install the dependencies.

Using `venv` or `conda`:
* Set up a virtual environment using `python -m venv venv` or `conda create -n myenv python=3.12`.
* Activate the virtual environment using `source venv/bin/activate` or `conda activate myenv`.
* Use `pip install -r requirements.txt` to install the dependencies.

Environment variables should be set in a `.env` file in the root directory of this repository.  The following environment variables are required:
* `ANTHROPIC_API_KEY`: Your Anthropic API key.
* `ANTHROPIC_BASE_URL`: The base URL for the Anthropic API (optional, defaults to `https://api.anthropic.com`).
* `DEFAULT_SYNTHETIC_MODEL`:  The model you wish to use for synthetic document generation (Claude 3.7 Sonnet is recommended).
* `DEFAULT_ANALYTIC_MODEL`:  The model you wish to use for topic discovery (Claude 4 Opus is recommended).
* `DEFAULT_ASSIGNMENT_MODEL`:  The model you wish to use for topic assignment (Claude 3.7 Sonnet is recommended).

