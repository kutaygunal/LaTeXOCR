# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security vulnerabilities. Report them privately so they can be addressed before disclosure.

To report a vulnerability, email the maintainer directly or open a private advisory on the repository's GitHub page. Include:

- A description of the vulnerability and its impact.
- Steps to reproduce, including any input image or command.
- The affected version and environment.

You will receive an acknowledgement within a reasonable time, and we will work with you to understand and fix the issue.

## Scope

LaTeXOCR is a local, offline tool. It reads image files and produces LaTeX text. It does not execute the target repository's code, does not make network calls except to a local Ollama instance when the AI engine is used, and does not read or print environment-file values.

## Supported versions

Security fixes are applied to the latest release. Please keep your installation up to date.
