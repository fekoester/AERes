# AERes

## Overview
AERes is a Python package for experimenting with the attention-enhanced reservoir architecture using PyTorch. This library includes modules to simulate dynamical systems like the Lorenz attractor, implement reservoir computing models, and apply linear attention mechanisms.

## Installation
You can install AERes using pip:

```bash
pip install aeres
```

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

### Simulating a Reservoir

Process the simulated data through a reservoir computing system to enhance its features for further analysis.

```python
from AERes.reservoir import Reservoir

# Create an instance of the Reservoir with specified parameters
reservoir = Reservoir(X_train, number_nodes=50, input_dimension=X_train.shape[1], seed=1, standardize=True)
```

### Training a Ridge Regression Model

Utilize a simple linear model for a baseline comparison.

```python
from sklearn.linear_model import Ridge

# Train a Ridge Regression model on the processed and standardized states
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(reservoir.states_stand, Y_train)

# Make predictions on the training data
ridge_predictions = ridge_model.predict(reservoir.states_stand)
```

### Implementing the Linear Attention Model

Set up and train a linear attention model to focus on important features dynamically.

```python
from AERes.attention import LinearAttentionTrainer

# Initialize the trainer for the Linear Attention model with the standardized reservoir states
trainer = LinearAttentionTrainer(reservoir.states_stand, Y_train, layer_type="linear")

# Train the model over 100 epochs
trainer.train(epochs=100)
```

### Evaluating the Model

Evaluate the performance of both models to understand their effectiveness.

```python
# Calculate the Mean Squared Error (MSE) for the Ridge Regression model
ridge_loss = trainer.MSELoss(ridge_predictions, Y_train)
print(f'Ridge Regression MSE for training: {ridge_loss.item()}')

# Evaluate the trained attention model on the testing data
trainer.evaluate(reservoir.states_stand, Y_test)
```

We can also make predictions on some given input and plot its outcome.

```python
import matplotlib.pyplot as plt
#Get the next step prediction for the testing set and plot it
pred_test = trainer.predict(reservoir.states_stand)
plt.plot(Y_test[:,0])
plt.plot(pred_test[:,0])
plt.show()
```

To perform a closed-loop configuration, we can use the one_step function of the reservoir system.

```python
#First run a few steps to converge the reservoir on the trajectory
N = 250
for i in range(N):
    reservoir.one_step(X_test[i], standardize=True)

#Then start the system and let it run autonomously
input = X_test[N]
prediction = []
for i in range(N):
    reservoir.one_step(input, standardize=True)
    #We have to reshape the reservoir state to the right tensor dimensions. Afterwards, we have to transpose and squeeze it
    pred_test = trainer.predict(reservoir.state_stand.reshape(1, -1)).T.squeeze() 
    prediction.append(pred_test)
    input = pred_test

plt.plot(np.array(prediction)[:,0])
plt.plot(Y_test[N:,0])
plt.show()
```

### Save the Model

To save time and computation, we can save the parameters of our model and load them for later use.

```python
#We can save and load the trained model under a specified path
trainer.save_model('path_to_model.pt')
trainer.load_model('path_to_model.pt')
```