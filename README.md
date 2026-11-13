# Incorporating Cognitive Load and Knowledge Transfer forMulti-Domain Knowledge Tracing

## Installation

### Requirements

- Python >= 3.7
- PyTorch >= 1.10.0
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the package directly:

```bash
pip install -e .
```

### Dependencies

- torch>=1.10.0
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- tqdm>=4.62.0

## Data Format

The training and test data should be in text format with comma-separated values:

- **Input format**: Each line represents a student's learning sequence
- **Data fields**: Knowledge concepts (KC), responses (R), questions (Q), question difficulties (QD), knowledge blocks (KB)

Additional required files:
- `{dataset}_graph.npz`: Graph structure representing knowledge relationships
- `{dataset}_qmatrix.json`: Question-to-knowledge concept mapping matrix

## Training the Model

### Basic Training

Navigate to the scripts directory and run:

```bash
cd scripts
python run_mkt.py --dataset_name junior --epoch 15 --lr 5e-3 --bs 16
```

### Training Parameters

#### Model Parameters

- `--n_q`: Number of questions (default: 170)
- `--n_kc`: Number of knowledge concepts (default: 139)
- `--n_kb`: Number of knowledge blocks (default: 50 for senior dataset)
- `--q_dim`: Question embedding dimension (default: 100)
- `--h_dim`: Hidden dimension (default: 90)
- `--lambda_p`: Lambda parameter for the model (default: 50)
- `--dropout`: Dropout rate (default: 0.2)

#### Training Parameters

- `--dataset_name`: Dataset name, either "junior" or "senior" (default: "junior")
- `--lr`: Learning rate (default: 5e-3)
- `--epoch`: Number of training epochs (default: 15)
- `--bs`: Batch size (default: 16)
- `--seq_len`: Sequence length (default: 25)

### GPU Configuration

By default, the model uses CUDA device 3 if available. To change the GPU device, modify the `device` variable in the training scripts:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Or set the environment variable:

```bash
export CUDA_VISIBLE_DEVICES=0
python run_mkt.py --dataset_name junior
```

Example output:

```
Epoch 1: Training --- loss : 0.45231, auc : 0.7234, accuracy : 0.6891, rmse : 0.3421
Epoch 1: Evaluating --- loss : 0.43129, auc : 0.7456, accuracy : 0.7012, rmse : 0.3287
valid auc improve: 0.0000 to 0.7456
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Reporting bugs
- Suggesting enhancements
- Submitting pull requests
- Code style and testing guidelines

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2026ltmkt,
  title     = {Incorporating Cognitive Load and Knowledge Transfer for Multi-Domain Knowledge Tracing},
  author    = {Zhang, Haotian and Wang, Shucun and Wu, Jinze and Ding, Liang and Liu, Shuochen and Huang, Zhenya and Sha, Jing and Wang, Shijin and Liu, Qi},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026},
  pages     = {6114--6125},
  doi       = {10.1145/3799682.3841120}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## Acknowledgments

This work builds upon advances in knowledge tracing, graph neural networks, and attention mechanisms in educational data mining.
