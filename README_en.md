# NACA0012 Airfoil Pin Fin Optimization Project

This project aims to find optimal pin fin structural parameters that maximize the ratio of heat transfer efficiency to flow resistance (Nu/(f^(1/3))) through computational fluid dynamics (CFD) and artificial intelligence (AI) techniques.

## Project Overview

Based on FiPy (a partial differential equation solver written in Python), this project simulates fluid flow and heat transfer processes around NACA0012 airfoils. By combining AI optimization algorithms, it automatically finds the optimal pin fin geometric parameters.

## Workflow

The optimization process consists of five stages:

### Stage 1: Toolchain Development
1. **Geometry Generation**: A C++ program generates airfoil coordinate files based on input parameters
2. **Mesh Generation**: Run `网格生成器2_三角形.py` to generate computational mesh
3. **CFD Solver**: Run `求解器_pvtnf.py` or `compute_Nu_f.py` for flow field and temperature field calculations

### Stage 2: Data Generation
1. **Sample Generation**: 
   ```bash
   python generate_samples.py
   ```
2. **Batch Simulation**:
   ```bash
   python run_all_cases.py
   ```

### Stage 3: AI Training
```bash
python train_ai_models.py
```

### Stage 4: Intelligent Optimization
```bash
python ai_optimization.py
```

### Stage 5: Final Validation
Run a complete CFD simulation with the optimal parameters found by AI for verification.

## Key Components

### Core Scripts
- `网格生成器2_三角形.py` - Mesh generation script
- `求解器_pvtnf.py` - CFD solver with heat transfer calculations
- `generate_samples.py` - Latin hypercube sampling for parameter combinations
- `run_all_cases.py` - Batch execution of all cases
- `train_ai_models.py` - Neural network training for Nu and f prediction
- `ai_optimization.py` - AI-based parameter optimization

### Key Parameters
- **Tt**: Pin fin top thickness parameter
- **Ts**: Pin fin side thickness parameter
- **Tad**: Pin fin leading edge angle parameter
- **Tb**: Pin fin trailing edge angle parameter

### Output Metrics
- **Nu**: Nusselt number, characterizing heat transfer efficiency
- **f**: Friction factor, characterizing flow resistance
- **Nu/(f^(1/3))**: Target parameter, the ratio of heat transfer efficiency to flow resistance

## Installation

### Prerequisites
- Python 3.7+
- FiPy
- Gmsh (for mesh generation)

### Setup
1. Create virtual environment:
   ```bash
   python -m venv myenvs_fipynaca2.0
   myenvs_fipynaca2.0\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Follow the five-stage workflow described above to perform the complete optimization process.

## Results

The project generates various outputs including:
- CFD simulation results
- Trained AI models (PyTorch)
- Optimization results
- Post-processing visualizations and reports

## License

This project is licensed under the MIT License - see the LICENSE file for details.