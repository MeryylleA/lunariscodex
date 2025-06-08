# Contributing to Lunaris Codex 🌙

First off, thank you for considering contributing to Lunaris Codex! We're excited to have you join our community. Whether you're fixing a bug, proposing a new feature, or improving the existing code, your help is invaluable.

This document provides guidelines for contributing to Lunaris Codex. These are mostly guidelines, not strict rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Development Setup](#development-setup)
- [Styleguides](#styleguides)
  - [Git Commit Messages](#git-commit-messages)
  - [Python Code Style](#python-code-style)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by the [Lunaris Codex Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs
If you find a bug in the training or inference scripts, please check the [GitHub Issues](https://github.com/MeryylleA/lunariscodex/issues) to see if it has already been reported. If not, please open a new issue and provide as much detail as possible, including:
- A clear, descriptive title.
- Steps to reproduce the bug.
- Full error messages and stack traces.
- Your environment (Python/PyTorch version, OS, CUDA version).

### Suggesting Enhancements
If you have an idea for a new feature or an improvement to the existing code, please open an issue to start a discussion. We value clear proposals that outline the potential benefits to the project.

### Submitting Pull Requests
When you're ready to contribute code:

1.  **Fork the repository** and create a new branch from `main`.
2.  **Make your changes.** Ensure your code adheres to the styleguides below.
3.  **Add or update tests** if you are adding new functionality.
4.  **Write clear commit messages.**
5.  **Open a Pull Request** against the `main` branch of the Lunaris Codex repository.
    *   Provide a clear description of the changes.
    *   Link to any relevant issues (e.g., "Closes #123").
    *   The project maintainer (@MeryylleA) will review your PR.

## Development Setup

1.  Clone your fork of the repository.
2.  Create and activate a Python virtual environment (Python 3.10+ recommended).
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    ```
3.  Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4.  To run tests and linters, you will need additional packages. You can install them manually:
    ```bash
    pip install pytest pytest-cov black flake8
    ```
5.  Run tests locally to ensure your changes haven't introduced regressions:
    ```bash
    pytest
    ```

## Styleguides

### Git Commit Messages
-   Use the present tense (e.g., "Add feature" not "Added feature").
-   We recommend using [Conventional Commits](https://www.conventionalcommits.org/) prefixes (e.g., `feat:`, `fix:`, `refactor:`, `test:`).
    *Example:* `feat: Add ReduceLROnPlateau scheduler to train.py`

### Python Code Style
-   Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
-   Format your code with [Black](https://github.com/psf/black) to maintain a consistent style.
-   Use a linter like [Flake8](https://flake8.pycqa.org/en/latest/) to catch common errors.
-   Add type hints to your Python code.

## Community

-   Join our [GitHub Discussions](https://github.com/MeryylleA/lunariscodex/discussions) for questions and ideas.
-   Be respectful and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

Thank you for helping make Lunaris Codex a better tool for everyone!
