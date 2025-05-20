# Security Policy for Lunaris Codex 🌙

## Reporting a Vulnerability

The Lunaris Codex project and its maintainer take security bugs in Lunaris Codex seriously. We appreciate your efforts to responsibly disclose your findings and will make every effort to acknowledge your contributions.

We are primarily interested in vulnerabilities within the core Lunaris Codex codebase, including:
-   The PyTorch model architecture (`model.py`).
-   The data preparation pipeline (`prepare_data.py`).
-   The training pipeline (`train.py`).
-   The inference script (`inference.py`).
-   The C++ utility tools (e.g., `bpe_trainer.cpp`, `lunaris_data_analyzer.cpp`, `lunaris_text_cleaner.cpp`).
-   The CI/CD workflow configurations (`.github/workflows/`).

**How to Report a Security Vulnerability:**

If you believe you have found a security vulnerability in Lunaris Codex, please **DO NOT open a public GitHub issue.** We ask that you report it privately to ensure that the vulnerability is not exploited before it can be addressed.

Please send an email to:
**[support@mooncloudservices.tech]**

**Please include the following details in your report:**

*   A clear and descriptive title for your report.
*   A detailed description of the vulnerability.
*   The component, file(s), or function(s) affected.
*   Clear steps to reproduce the vulnerability (if applicable, include code snippets, configurations, specific inputs, or links to gists).
*   The potential impact of the vulnerability if exploited (e.g., data leakage, code execution, denial of service).
*   Any suggested mitigations or fixes, if you have them.
*   Your name or alias for acknowledgment (if you wish to be credited), or a request to remain anonymous.

**What to Expect After Reporting:**

*   We will acknowledge receipt of your vulnerability report, typically within 48-72 hours.
*   We will investigate the report to determine if it's a valid security vulnerability within the scope of this policy.
*   For valid vulnerabilities, we will work to address them in a timely manner and will aim to communicate an estimated timeframe for a fix where possible.
*   We will keep you informed of our progress as appropriate.
*   Once a vulnerability is fixed and a new version is released, we aim to credit responsible reporters in the release notes or commit messages, unless you prefer to remain anonymous.

**Scope - What is considered a security vulnerability for Lunaris Codex?**

*   Bugs in our Python or C++ code that could lead to:
    *   Arbitrary code execution when processing untrusted inputs (e.g., specially crafted dataset files, malicious model configuration files if loaded, or potentially via deserialization of untrusted checkpoint files).
    *   Denial of Service (DoS) vulnerabilities that are more severe than typical program crashes (e.g., a DoS that could be triggered by a crafted input and significantly degrade system resources, or one that corrupts critical data).
    *   Security issues in our C++ utilities (e.g., buffer overflows when parsing files, command injection if arguments are mishandled in a specific way, memory corruption).
    *   Significant flaws in the randomness or security-critical aspects of any helper functions (e.g., insecure generation of temporary files, predictable seeds in security-sensitive contexts, though Lunaris Codex is not a cryptographic library).
*   Vulnerabilities in the GitHub Actions workflows that could compromise the integrity of the build process, lead to secret exposure, or allow unauthorized modifications.

**Out of Scope for this Policy (but still welcome as general bug reports/issues):**

*   Bugs that primarily cause the training to produce suboptimal results (e.g., high loss, low accuracy) unless they stem from an underlying security flaw.
*   General performance issues or standard program crashes due to incorrect usage or invalid inputs that do not have a clear security exploitation vector.
*   Vulnerabilities in third-party dependencies (e.g., PyTorch, Transformers, NumPy). These should be reported directly to the maintainers of those libraries. However, if Lunaris Codex *uses a dependency in a demonstrably insecure way*, that *could* be in scope.
*   Security aspects of external datasets linked or exampled (e.g., the content of `Lunaris-Data` itself, if it contained malicious example snippets not intended as such). This is primarily a data quality/curation concern.
*   Social engineering, phishing, or security of the GitHub platform itself.

**Responsible Disclosure Philosophy:**

We believe in and practice responsible disclosure. Please give us a reasonable amount of time to investigate and address reported vulnerabilities before publicly disclosing them. We are committed to working with security researchers to make Lunaris Codex as secure as possible.

Thank you for helping keep Lunaris Codex secure!
