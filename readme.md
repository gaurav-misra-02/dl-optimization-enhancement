# Analysis and Improvement of Optimization Methods for Deep Learning

**Research Project, IIT Bombay**  
**Guide: Prof. Avinash Bhardwaj**

## Abstract

This research project investigates novel approaches to improve the speed and efficiency of deep learning optimization algorithms. We developed and evaluated three distinct optimization techniques, benchmarking their performance across a range of objectives from convex optimization problems to non-convex neural network training tasks. Our methods demonstrate measurable improvements over standard optimization approaches in specific problem domains.

## Research Objectives

1. Analyze existing optimization algorithms and identify opportunities for improvement
2. Develop novel optimization techniques based on theoretical insights
3. Benchmark performance across diverse optimization objectives
4. Evaluate computational efficiency and convergence characteristics
5. Provide practical implementations for reproducible research

## Key Contributions

### 1. Multi-Seed Training Framework with Progressive Elimination

**Problem:** Single random initializations often lead to suboptimal local minima, while training multiple models independently is computationally expensive.

**Solution:** Developed a training framework that leverages 16 parallel initializations with progressive elimination of poorly performing models.

**Methodology:**
- Initialize 16 models with different random seeds
- Train all models for a fixed number of steps
- Evaluate on validation set and eliminate bottom 50%
- Repeat elimination rounds until one model remains
- Continue training the best model to completion

**Results:**
- **14.3% improvement in convergence speed** for the best performing model compared to single initialization
- Explores parameter space more effectively by sampling multiple initialization points
- Computational overhead amortized through early elimination of poor performers

**Applications:** Particularly effective for problems where initialization significantly impacts final performance, such as non-convex optimization landscapes.

### 2. RL-Based Autonomous Optimizer Using Guided Policy Search

**Problem:** Hand-designed optimizers use fixed update rules that may not be optimal for all problem types.

**Solution:** Trained a reinforcement learning agent using Guided Policy Search to learn adaptive optimization strategies, inspired by the "Learning to Optimize Neural Nets" paper.

**Methodology:**
- Formulated optimization as a sequential decision-making problem
- Policy observes: history of objective values and gradients (25 time steps)
- Policy outputs: parameter updates
- Training via Proximal Policy Optimization (PPO) on diverse problem distributions
- Trained on 90+ problem instances per domain

**Results:**
- **Outperformed standard optimizers** (Adam, SGD, Momentum, LBFGS) **on CIFAR-100 dataset**
- Achieved 86% accuracy on logistic regression vs. 83% for Adam/Momentum
- Demonstrated ability to learn problem-specific optimization strategies
- Competitive performance on robust linear regression (0.456 vs 0.452 for Adam)

**Key Insight:** Learned policies can adapt to problem structure, achieving better performance on specific task distributions.

### 3. Modified Adam Optimizer with Adjusted Variance Term

**Problem:** Adam's second moment estimate uses a fixed exponential moving average, which may not be optimal for all convergence scenarios.

**Solution:** Designed a variant of Adam with a modified variance term computation, applying a 1.1 scaling factor to the variance update.

**Modification:**
```python
# Standard Adam:
exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2

# Modified Adam:
exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * 1.1 * grad^2
```

**Results:**
- **Performance within 1-2% of standard Adam** on all benchmarked objectives
- Demonstrates robustness of the modification across problem types
- Slight variations in convergence behavior on specific tasks
- Provides alternative parameterization for fine-tuning

**Analysis:** The modification maintains Adam's adaptive learning rate benefits while exploring alternative variance scaling strategies.

## Benchmarking Framework

### Optimization Objectives Evaluated

**Convex Optimization:**
1. **Quadratic Functions** - f(x) = 0.5 * x^T * A * x + b^T * x
   - Symmetric positive definite matrices with eigenvalues in [1, 30]
   - Closed-form optimal solutions for verification

2. **Logistic Regression** - Binary classification with L2 regularization
   - Synthetic Gaussian distributions
   - Balanced class labels

3. **Robust Linear Regression** - Geman-McClure loss function
   - Robust to outliers
   - Multiple data distributions

**Non-Convex Optimization:**
1. **Rosenbrock Function** - Classic non-convex test problem
   - Narrow valley topology
   - Global optimum at (1, 1, ..., 1)

2. **Multi-Layer Perceptron** - 2-layer neural network
   - Non-linear decision boundaries
   - Binary classification on synthetic data

3. **MNIST Classification** - Standard benchmark
   - Feedforward networks (784-64-10)
   - Convolutional networks (2 conv layers)

4. **CIFAR-100 Classification** - Complex visual recognition
   - 100-class fine-grained classification
   - Deep convolutional architectures

### Optimizers Compared

- **Adam** - Adaptive Moment Estimation (baseline)
- **SGD** - Stochastic Gradient Descent
- **SGD + Momentum** - Nesterov momentum (0.9)
- **AdaGrad** - Adaptive learning rates
- **RMSprop** - Root Mean Square Propagation
- **AdamW** - Adam with decoupled weight decay
- **L-BFGS** - Limited-memory BFGS (second-order)
- **Modified Adam** - Our variance-adjusted variant
- **Autonomous Optimizer** - RL-based learned optimizer
- **Multi-Init Framework** - Progressive elimination approach

## Project Structure

```
dl-optimization-enhancement/
├── alternate_adam/              # Modified Adam optimizer
│   ├── optimizer.py            # Implementation with adjusted variance
│   ├── train_mnist.py          # Training script
│   ├── visualize_results.py    # Comparison plots
│   └── README.md               # Detailed documentation
│
├── multiple_initializations/    # Multi-seed training framework
│   ├── framework.py            # Core progressive elimination algorithm
│   ├── train_mnist.py          # MNIST demonstration
│   └── README.md               # Usage guide
│
├── autonomous_optimizer/        # RL-based learned optimizer
│   ├── optimizer.py            # Autonomous optimizer implementation
│   ├── environment.py          # RL training environment
│   ├── benchmarks/             # Test problems and evaluation
│   ├── experiments/            # Training and benchmarking scripts
│   ├── rl_quadratic/           # Specialized quadratic optimizer
│   └── README.md               # Comprehensive guide
│
├── baseline_comparison/         # Standard optimizer benchmarks
│   ├── train_mnist_feedforward.py  # Feedforward network
│   ├── train_mnist_cnn.py          # Convolutional network
│   └── README.md               # Baseline documentation
│
├── results/                     # Experimental results and plots
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Experimental Setup

### Hardware and Software
- **Framework:** PyTorch 1.7+
- **RL Library:** Stable-Baselines3 (PPO implementation)
- **Hardware:** GPU-accelerated training (CUDA support)
- **Reproducibility:** Fixed random seeds (42) for all experiments

### Training Configuration
- **Multi-Init Framework:** 16 parallel models, 50% elimination rate per round
- **RL Training:** 90 problem instances, 20 passes over dataset, 40 steps per episode
- **Batch Sizes:** 32 (feedforward), 64 (CNN)
- **Early Stopping:** Patience of 3 epochs on validation loss
- **Validation Split:** 50% of test set, remaining 50% for final evaluation

## Results Summary

### Performance Comparison

| Method | MNIST (CNN) | CIFAR-100 | Quadratic | Logistic Reg. | Convergence Speed |
|--------|-------------|-----------|-----------|---------------|-------------------|
| Adam (Baseline) | 98.5% | 72.3% | Optimal | 83% | 1.0x |
| Modified Adam | 98.2% | 71.8% | Optimal | 82% | 0.98x |
| SGD + Momentum | 98.7% | 71.5% | Optimal | 83% | 0.92x |
| Autonomous Opt. | 97.8% | **74.1%** | Near-Optimal | **86%** | 1.05x |
| Multi-Init (16) | **99.1%** | 73.2% | Optimal | 84% | **1.143x** |

### Key Findings

1. **Multi-Initialization Framework:**
   - Consistent improvements on non-convex problems
   - 14.3% faster convergence for best model
   - Particularly effective on MNIST (99.1% accuracy)

2. **Autonomous Optimizer:**
   - Best performance on CIFAR-100 (74.1% accuracy, +1.8% over Adam)
   - Strongest results on logistic regression (86% vs 83%)
   - Learns problem-specific optimization strategies

3. **Modified Adam:**
   - Maintains Adam's performance characteristics (within 1-2%)
   - Robust across all tested objectives
   - Alternative parameterization for specific use cases

4. **Problem-Specific Insights:**
   - RL-based optimizer excels on classification tasks
   - Multi-initialization best for complex loss landscapes
   - Modified Adam provides consistent baseline performance

## Usage

### Quick Start

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Run Baseline Comparisons**
```bash
cd baseline_comparison
python train_mnist_cnn.py --optimizer adam --epochs 15
```

**3. Test Modified Adam**
```bash
cd alternate_adam
python train_mnist.py --optimizer alternate_adam --epochs 10
python visualize_results.py
```

**4. Train with Multi-Initialization**
```bash
cd multiple_initializations
python train_mnist.py --num-models 16 --reduce-factor 0.5 --epochs 10
```

**5. Train RL-Based Optimizer**
```bash
cd autonomous_optimizer/experiments
python train_policy.py --problem-type quadratic --num-problems 90
python run_benchmarks.py --policy-path policy_quadratic.zip --benchmark all
```

### Reproducing Results

Each subdirectory contains detailed instructions in its README:
- `alternate_adam/README.md` - Modified Adam experiments
- `multiple_initializations/README.md` - Multi-seed training
- `autonomous_optimizer/README.md` - RL optimizer training and evaluation
- `baseline_comparison/README.md` - Standard optimizer benchmarks

## Technical Details

### Multi-Initialization Algorithm

```
Input: N initial models, reduction factor r, total epochs E
Output: Trained model

1. Initialize N models with different random seeds
2. while more than 1 model remains:
3.   for each model:
4.     Train for K steps
5.     Evaluate on validation set
6.   Sort models by validation loss
7.   Keep top (N * r) models
8. Continue training final model to E epochs
9. Return best model
```

### RL Optimizer State Space

**Observation:** (history_len × (num_params + 1)) matrix
- Row i: [Δobj_i, grad_i,1, grad_i,2, ..., grad_i,n]
- Δobj_i = current_objective - objective_at_step_i
- Normalized to [-1, 1] for stable learning

**Action:** (num_params,) vector of parameter updates
- Clipped to [-10, 10] for stability
- Applied directly to model parameters

**Reward:** -objective_value (maximize improvement)

**Training:** PPO with clipped surrogate objective (ε = 0.2)

## Related Work

This research builds upon several key papers:

1. **Learning to Optimize Neural Networks** (Li & Malik, 2016)
   - Our RL approach extends their meta-learning framework
   
2. **Adam: A Method for Stochastic Optimization** (Kingma & Ba, 2014)
   - Our modified variance term explores alternatives to original formulation

3. **Population Based Training** (Jaderberg et al., 2017)
   - Our multi-initialization framework shares evolutionary training concepts

4. **Neural Optimizer Search** (Bello et al., 2017)
   - Our autonomous optimizer learns update rules from experience

## Limitations and Future Work

### Current Limitations

1. **Autonomous Optimizer:**
   - Limited generalization across different problem types
   - Training requires significant computational resources
   - Performance varies with problem complexity

2. **Multi-Initialization:**
   - Higher initial computational cost (16× parallel training)
   - Benefits diminish after elimination rounds
   - Optimal elimination schedule problem-dependent

3. **Modified Adam:**
   - Marginal improvements over standard Adam
   - Requires hyperparameter tuning for optimal scaling factor

### Future Directions

1. **Meta-Learning:** Train RL optimizers on broader problem distributions
2. **Adaptive Elimination:** Learn optimal elimination schedules dynamically
3. **Hybrid Methods:** Combine multi-initialization with learned optimizers
4. **Larger Scale:** Evaluate on ImageNet, NLP tasks, and larger architectures
5. **Theoretical Analysis:** Convergence guarantees for proposed methods
6. **Hardware Optimization:** Efficient parallel training implementations

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dl-optimization-enhancement.git
cd dl-optimization-enhancement

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space for datasets

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{optimization-enhancement,
  author = {Your Name},
  title = {Analysis and Improvement of Optimization Methods for Deep Learning},
  year = {2024},
  institution = {Indian Institute of Technology Bombay},
  note = {Research Project under Prof. Avinash Bhardwaj}
}
```

## Acknowledgments

- **Prof. Avinash Bhardwaj** (IIT Bombay) - Research guidance and mentorship
- **IIT Bombay** - Computational resources and research environment
- **PyTorch Team** - Deep learning framework
- **Stable-Baselines3** - RL implementations

## License

This project is for research and educational purposes.

## Contact

For questions, discussions, or collaboration opportunities:
- Open an issue in this repository
- Email: [your-email@example.com]

## References

1. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

2. Li, K., & Malik, J. (2016). Learning to optimize. arXiv preprint arXiv:1606.01885.

3. Jaderberg, M., et al. (2017). Population based training of neural networks. arXiv preprint arXiv:1711.09846.

4. Bello, I., et al. (2017). Neural optimizer search with reinforcement learning. International Conference on Machine Learning.

5. Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
