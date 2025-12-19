# Security Policy

## Supported Versions

| Version | Supported          |
|:--------|:-------------------|
| 0.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in imgaug2, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email the maintainers directly or use GitHub's private vulnerability reporting feature:

1. Go to the [Security tab](https://github.com/Alfredo-Sandoval/imgaug2/security)
2. Click "Report a vulnerability"
3. Provide a detailed description of the issue

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 7 days
- **Resolution target**: Within 30 days for critical issues

## Security Considerations

imgaug2 processes image data and executes augmentation pipelines. When using this library:

- **Input validation**: Always validate image inputs from untrusted sources
- **File paths**: Be cautious with user-provided file paths to prevent path traversal
- **Pickle files**: Never load augmenter configurations from untrusted pickle files
- **Resource limits**: Large images or complex pipelines may consume significant memory

## Dependencies

imgaug2 depends on several third-party libraries. We monitor for vulnerabilities in our dependency chain and update promptly when security patches are available.
