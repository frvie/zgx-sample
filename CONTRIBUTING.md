# Contributing to Unsloth Llama Fine-tuning

Thank you for your interest in contributing! 🎉

## How to Contribute

### 🐛 Bug Reports
- Use the issue tracker to report bugs
- Include system information (GPU, CUDA version, Python version)
- Provide reproduction steps and error messages

### ✨ Feature Requests  
- Describe the feature and use case
- Consider if it fits the project's scope
- Check if it's already requested in issues

### 🔧 Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Update documentation if needed
6. Submit a pull request

## Development Setup

```bash
# Clone the repository
git clone <your-fork-url>
cd unsloth

# Set up environment
source .venv/bin/activate
uv sync

# Test the setup
python unsloth-finetune-llama3_2-1B.py --help
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Include type hints where helpful

## Testing

- Test your changes with small datasets first
- Verify GPU memory usage
- Check that training completes successfully
- Test both training and inference

## Areas for Contribution

- 📊 Additional validation metrics
- 🎯 Support for more model architectures
- 📈 Enhanced plotting and visualization
- 🔧 Performance optimizations
- 📚 Documentation improvements
- 🧪 Unit tests

## Questions?

Feel free to open an issue for questions or discussion!

Thank you for helping make this project better! 🚀