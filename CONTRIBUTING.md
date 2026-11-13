# Contributing to LT-MKT

Thank you for your interest in contributing to LT-MKT!

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, PyTorch version, OS)

### Suggesting Enhancements

We welcome feature requests! Please open an issue describing:
- The motivation for the enhancement
- How it would benefit users
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure code follows existing style
6. Update documentation as needed
7. Commit with clear messages (`git commit -m 'Add amazing feature'`)
8. Push to your fork (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

### Testing

- Test your changes on both CPU and GPU if possible
- Verify the model trains without errors
- Check that metrics (AUC, accuracy, RMSE) are reasonable
- Test on both junior and senior datasets if available

### Documentation

- Update README.md if adding new features
- Add docstrings for new functions
- Include usage examples for new functionality
- Update requirements.txt if adding dependencies

## Areas for Contribution

We especially welcome contributions in:

### 1. Baseline Models
Implement additional baseline models for comparison:
- SAKT (Self-Attentive Knowledge Tracing)
- CKT (Convolutional Knowledge Tracing)
- SAINT (Separated Self-AttentIve Neural Knowledge Tracing)
- GKT (Graph-based Knowledge Tracing)

### 2. Data Processing
- Support for additional dataset formats
- Data augmentation techniques
- Automated data validation tools

### 3. Graph Construction
- Alternative graph construction methods
- Graph refinement techniques
- Visualization of learned attention weights

### 4. Model Improvements
- Additional attention mechanisms
- Multi-task learning extensions
- Transfer learning capabilities
- Interpretability tools

### 5. Evaluation
- Additional evaluation metrics
- Cross-dataset evaluation
- Ablation study tools

### 6. Documentation
- Tutorial notebooks
- Video tutorials
- Translation to other languages
- API documentation

## Questions?

Feel free to open an issue for any questions about contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming community for everyone.
