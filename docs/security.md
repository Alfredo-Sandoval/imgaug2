# Security

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainers directly
3. Include details about the vulnerability
4. Allow time for a fix before public disclosure

## Dependencies

imgaug2 depends on several image processing libraries:

- NumPy
- OpenCV
- SciPy
- scikit-image
- Pillow

Keep these dependencies updated to receive security patches.

## Image Processing Safety

When processing untrusted images:

- Validate image dimensions before processing
- Set reasonable limits on image sizes
- Handle exceptions from corrupted files
