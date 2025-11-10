# Neural ODE Demonstration

This project demonstrates the training of **Neural Ordinary Differential Equations (Neural ODEs)** on a simple spiral dynamical system.

## 📚 What are Neural ODEs?

Neural ODEs replace discrete layers in neural networks with continuous transformations described by ordinary differential equations:

```
dh(t)/dt = f_θ(h(t), t)
```

where `f_θ` is a neural network with parameters `θ`.

### Key Advantages:
- **Memory-efficient training** using the adjoint method (O(1) memory)
- **Adaptive computation** - can evaluate at any time point
- **Natural modeling** of continuous-time processes
- **Fewer parameters** compared to traditional recurrent architectures

## 🚀 Getting Started

### Prerequisites

You need Python 3.7+ with the following packages:
- PyTorch
- torchdiffeq
- matplotlib
- numpy

### Installation

Install the required packages:

```bash
pip install torch torchdiffeq matplotlib numpy
```

### Running the Notebook

1. Open `neural_ode_demo.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially
3. The notebook will:
   - Generate spiral trajectory data
   - Train a Neural ODE to learn the dynamics
   - Visualize training progress
   - Compare learned vs true dynamics
   - Test extrapolation capabilities

## 📊 What the Notebook Demonstrates

### 1. **Data Generation**
Creates a spiral trajectory from a known ODE system with initial condition `[2.0, 0.0]`

### 2. **Model Architecture**
Builds a Neural ODE with:
- Input: 2D state (x, y)
- Hidden layer: 50 neurons with Tanh activation
- Output: 2D dynamics

### 3. **Training**
- Uses the adjoint method for memory-efficient backpropagation
- Mini-batch training with random trajectory subsequences
- Adam optimizer with learning rate 1e-3
- 2000 training iterations

### 4. **Evaluation**
- Compares learned dynamics with true dynamics
- Visualizes trajectories, phase portraits, and vector fields
- Tests extrapolation beyond training time range

## 📈 Expected Results

After training, you should see:
- **Training loss** decreasing from ~0.1 to <0.001
- **Phase portrait** matching the true spiral closely
- **Vector field** showing the learned dynamics
- **Good extrapolation** beyond the training time range (0-25s)

## 🔬 Key Concepts Illustrated

1. **Continuous Depth Networks**: ODEs provide infinite-depth networks
2. **Adjoint Sensitivity Method**: Efficient backpropagation through ODE solvers
3. **Adaptive Computation**: Variable-step ODE solvers adjust based on dynamics
4. **Time-Continuous Models**: Natural representation for physical systems

## 📖 References

- **Original Paper**: [Neural Ordinary Differential Equations (NeurIPS 2018)](https://arxiv.org/abs/1806.07366)
  - Chen, Ricky T. Q., et al.
  
- **Library**: [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
  - PyTorch implementation of differentiable ODE solvers

## 🎯 Potential Applications

- Time series forecasting
- Physics-informed machine learning
- Continuous normalizing flows (generative models)
- Robotics and control systems
- Medical signal processing (EEG, ECG)
- Climate modeling
- Chemical kinetics

## 🛠️ Customization

You can modify the notebook to:
- Change the true dynamics (modify `TrueDynamics` class)
- Adjust network architecture (hidden dimensions, activation functions)
- Try different ODE solvers (`dopri5`, `rk4`, `adaptive_heun`, etc.)
- Experiment with different initial conditions
- Extend to higher-dimensional systems

## 📝 Notes

- **GPU Support**: The code automatically uses GPU if available
- **Training Time**: Approximately 2-5 minutes on CPU, <1 minute on GPU
- **Reproducibility**: Random seeds are set for reproducible results
- **Visualization**: All plots are generated inline in the notebook

## 🤝 Contributing

Feel free to experiment with different:
- Dynamical systems (pendulum, Lorenz attractor, etc.)
- Network architectures
- Training strategies
- Visualization techniques

## 📄 License

This educational project is provided as-is for learning purposes.

## 🙋 Troubleshooting

**Issue**: Module 'torchdiffeq' not found
- **Solution**: Run `pip install torchdiffeq`

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` or use CPU

**Issue**: Poor convergence
- **Solution**: Try lower learning rate or more iterations

---

Happy learning! 🎓