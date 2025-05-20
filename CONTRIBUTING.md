# Contributing to Lunaris Codex 🌙

First off, thank you for considering contributing to Lunaris Codex! We're excited to have you join our community. Whether you're fixing a bug, proposing a new feature, improving documentation, or sharing your experiments, your help is invaluable.

This document provides a set of guidelines for contributing to Lunaris Codex, which is hosted on GitHub at [https://github.com/MeryylleA/lunariscodex](https://github.com/MeryylleA/lunariscodex). These are mostly guidelines, not strict rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements or New Features](#suggesting-enhancements-or-new-features)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Styleguides](#styleguides)
  - [Git Commit Messages](#git-commit-messages)
  - [Python Styleguide](#python-styleguide)
  - [C++ Styleguide (for utilities)](#c-styleguide-for-utilities)
  - [Documentation Styleguide](#documentation-styleguide)
- [Testing](#testing)
- [Community and Discussions](#community-and-discussions)

## Code of Conduct

This project and everyone participating in it is governed by the [Lunaris Codex Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior.

## How Can I Contribute?

There are many ways to contribute to Lunaris Codex:

### Reporting Bugs

If you encounter a bug, please help us by submitting an issue to our [GitHub Repository](https://github.com/MeryylleA/lunariscodex/issues). Before creating a bug report, please check existing issues to see if the problem has already been reported.

When you are creating a bug report, please include as many details as possible. Fill out the "Bug Report" issue template if available. Key information includes:
- A clear and descriptive title.
- Steps to reproduce the bug.
- What you expected to happen.
- What actually happened (including full error messages and stack traces).
- Your environment (e.g., Python version, PyTorch version, OS, CUDA version if applicable, hardware).
- The specific version/commit hash of Lunaris Codex you are using.

### Suggesting Enhancements or New Features

We love to hear new ideas! If you have a suggestion for an enhancement or a new feature, please:
1. Check if there's an existing issue or discussion about your idea.
2. If not, open a new issue using the "Feature Request" template (if available) or start a new discussion in the "Ideas" category on our [GitHub Discussions page](https://github.com/MeryylleA/lunariscodex/discussions).
3. Clearly describe the proposed enhancement, its potential benefits, and any implementation ideas you might have.

### Your First Code Contribution

Unsure where to begin contributing to Lunaris Codex?
- Look for issues tagged `good first issue` or `help wanted`. These are usually tasks that are more self-contained and suitable for new contributors.
- Start by improving documentation, adding more tests, or fixing small, well-defined bugs.
- Feel free to ask questions in the relevant issue or on the Discussions page if you need guidance.

### Pull Requests

When you're ready to contribute code or documentation:

1.  **Fork the Repository:** Create your own copy of the `lunariscodex` repository on GitHub.
2.  **Create a Branch:** Create a new branch in your fork for your changes (e.g., `git checkout -b feature/my-new-feature` or `fix/bug-description`).
3.  **Develop & Test:** Make your changes, and ensure you add or update tests as appropriate. We encourage following the styleguides (see below), especially for Python.
4.  **Commit Your Changes:** Write clear, concise commit messages (see [Git Commit Messages](#git-commit-messages)).
5.  **Push to Your Fork:** `git push origin your-branch-name`.
6.  **Open a Pull Request (PR):**
    *   Go to the Lunaris Codex repository and you should see a prompt to open a PR from your new branch.
    *   Fill out the PR template (if available) with a clear description of your changes, why they are needed, and how they were tested.
    *   Link to any relevant issues (e.g., "Closes #123").
    *   Ensure your PR passes all automated CI checks (GitHub Actions).
    *   The project maintainer (@MeryylleA) will review your PR. Be prepared to discuss your changes and make further adjustments if requested.

## Development Setup

1.  Clone your fork of the repository.
2.  Create a Python virtual environment (e.g., Python 3.10+ recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate 
    ```
3.  Install dependencies:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    # For development (tests, linters), consider installing:
    # pip install pytest pytest-cov black flake8
    ```
4.  For the C++ utilities (e.g., `text_cleaner`, `data_analyzer`, `bpe_trainer`), you'll need a C++17 compiler (like `g++`). They can be compiled using the main `Makefile` in the root of the project (e.g., `make all` or `make <utility_name>`), or by following instructions in their respective `README.md` files.

## Styleguides

### Git Commit Messages

-   Use the present tense ("Add feature" not "Added feature").
-   Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
-   Limit the first line to 72 characters or less.
-   Reference issues and PRs liberally.
-   Consider using [Conventional Commits](https://www.conventionalcommits.org/) prefixes like `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`.
    *Example:* `feat: Add support for text_file_chunks in prepare_data.py`

### Python Styleguide

-   Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
-   We recommend using a code formatter like [Black](https://github.com/psf/black) to ensure consistent style.
-   We recommend using a linter like [Flake8](https://flake8.pycqa.org/en/latest/) to catch errors and style issues.
-   Add type hints to your Python code where it improves clarity and maintainability.

### C++ Styleguide (for utilities)

-   Aim for modern C++ (C++17 is used for our utilities).
-   Follow a consistent style. The [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) can be a good reference, but a simpler, internally consistent style is the primary goal.
-   Use comments to explain complex parts of the code.

### Documentation Styleguide

-   Use Markdown for READMEs and other documentation files.
-   Write clear and concise English.
-   For Python docstrings, consider following [PEP 257](https://www.python.org/dev/peps/pep-0257/) and a common style like Google's or NumPy's.

## Testing

-   **Python:** We aim to increase test coverage. If you add new features or fix bugs, please add or update corresponding tests (using `pytest`). Place tests in the `tests/` directory.
-   **C++ Utilities:** Basic command-line execution tests are part of the CI. More extensive tests can be added as needed.
-   **CI:** Our primary [GitHub Actions workflow](.github/workflows/ci.yml) runs a comprehensive suite of tests, including Python pipeline E2E tests, C++ utility compilation and basic execution, and Python unit tests with coverage reporting. Ensure your changes pass these tests.

## Community and Discussions

-   Join our [GitHub Discussions](https://github.com/MeryylleA/lunariscodex/discussions) page for questions, ideas, and to show off what you're building!
-   Be respectful and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

Thank you for contributing to making Lunaris Codex a better tool for everyone!
