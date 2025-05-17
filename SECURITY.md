# Security Policy for Lunaris Codex

## Reporting a Vulnerability

The Lunaris Codex team and community take security bugs in Lunaris Codex seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

We are primarily interested in vulnerabilities within the core Lunaris Codex codebase:
-   The PyTorch model architecture (`model.py`).
-   The data preparation pipeline (`prepare_data.py`).
-   The training pipeline (`train.py`).
-   The C++ utility tools (e.g., `lunaris_data_analyzer.cpp`, `lunaris_text_cleaner.cpp`).
-   The CI workflow configurations (`.github/workflows/`).

**How to Report a Security Vulnerability:**

If you believe you have found a security vulnerability in Lunaris Codex, please **DO NOT open a public GitHub issue.** We ask that you report it privately to ensure that the vulnerability is not exploited before it can be addressed.

Please send an email to:
**[support@mooncloudservices.tech]**

**Please include the following details in your report:**

*   A clear description of the vulnerability.
*   The component or file(s) affected.
*   The steps to reproduce the vulnerability (if applicable, include code snippets, configurations, or specific inputs).
*   The potential impact of the vulnerability if exploited.
*   Any suggested mitigations or fixes, if you have them.
*   Your name or alias for acknowledgment (if you wish to be credited).

**What to Expect:**

*   We will acknowledge receipt of your vulnerability report, typically within 48-72 hours.
*   We will investigate the report and determine if it's a valid security vulnerability within the scope of this policy.
*   We will work to address valid vulnerabilities in a timely manner.
*   We will keep you informed of our progress.
*   Once a vulnerability is fixed, we aim to credit responsible reporters in the release notes or commit messages, unless you prefer to remain anonymous.

**Scope - What is considered a security vulnerability for Lunaris Codex?**

*   Bugs in our code that could lead to:
    *   Arbitrary code execution when processing untrusted data (e.g., specially crafted dataset files, malicious model configuration files if we support loading them).
    *   Denial of Service (DoS) vulnerabilities that are more severe than typical program crashes (e.g., a DoS that could be triggered remotely if Lunaris Codex were part of a service, or one that corrupts critical data).
    *   Security issues in our C++ utilities (e.g., buffer overflows when parsing files, command injection if arguments are mishandled in a specific way).
    *   Significant flaws in the randomness or security-critical aspects of any cryptographic-like helper functions (though Lunaris Codex itself is not a cryptographic library, it might use random numbers for initializations, etc.).
*   Vulnerabilities in the GitHub Actions workflows that could compromise the integrity of the build process or expose secrets (though we strive to use best practices here).

**Out of Scope for this Policy (but still welcome as general bug reports/issues):**

*   Bugs that cause the training to produce suboptimal results (e.g., high loss, low accuracy) unless they stem from a clear security flaw.
*   Performance issues that are not security-related.
*   Standard program crashes due to incorrect usage or invalid inputs that do not have a clear security exploitation vector.
*   Vulnerabilities in third-party dependencies (e.g., PyTorch, Transformers, NumPy). These should be reported to the maintainers of those libraries. However, if Lunaris Codex uses a dependency in an insecure way, that *could* be in scope.
*   Security of the [Lunaris-Data dataset](https://huggingface.co/datasets/meryyllebr543/lunaris-data) itself (e.g., if the data *contained* malicious code snippets that were intended to be benign). This is more of a data quality/curation issue, though interesting.

**Responsible Disclosure Philosophy:**

We believe in responsible disclosure. Please give us a reasonable amount of time to address reported vulnerabilities before publicly disclosing them. We are committed to working with security researchers to make Lunaris Codex as secure as possible.

Thank you for helping keep Lunaris Codex secure!
