# AERes

## Overview
AERes is a Python package for experimenting with the attention-enhanced reservoir architecture using PyTorch. This library includes modules to simulate dynamical systems like the Lorenz attractor, implement reservoir computing models, and apply linear attention mechanisms.

## Installation
You can install AERes using pip:

```bash
pip install aeres


## Quick Start

This guide will help you get started with the `AERes` library, demonstrating how to simulate a dynamical system, utilize reservoir computing, and apply both Ridge Regression and a linear attention model for analysis.

### Simulating a Dynamical System

Begin by simulating a coupled Lorenz system, which is a common example of a chaotic dynamical system.

```python
from AERes.dynamicalSystems import LorenzSimulator

# Initialize the Lorenz simulator with the 'lorenz_coupled' configuration
lorenz_simulator = LorenzSimulator(function_name='lorenz_coupled')

# Perform the simulation and retrieve the data split into training and testing sets
X_train, Y_train, X_test, Y_test = lorenz_simulator.simulate_lorenz()
```