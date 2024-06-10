from sklearn.linear_model import Ridge
from AERes.dynamicalSystems import LorenzSimulator
from AERes.reservoir import Reservoir
from AERes.attention import LinearAttentionTrainer

import matplotlib.pyplot as plt
import numpy as np

#Simulate a Lorenz as a basic example for a chaotic dynamical system
lorenz_simulator = LorenzSimulator(function_name='lorenz')
X_train, Y_train, X_test, Y_test = lorenz_simulator.simulate_lorenz()

# Create an instance of the Reservoir, feed it the input and run it
reservoir = Reservoir(X_train, number_nodes=50, input_dimension=X_train.shape[1], seed=1, standardize=True)

# Train a simple Ridge Regression Model for comparison
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(reservoir.states_stand, Y_train)
ridge_predictions = ridge_model.predict(reservoir.states_stand)

# Create a linear attention model
trainer = LinearAttentionTrainer(reservoir.states_stand, Y_train, layer_type="linear")

#Check the training and testing error for the ridge regression with the implemented attention model error function
loss = trainer.MSELoss(ridge_predictions, Y_train)
print(f'Ridge Regression MSE for training: {loss.item()}')
trainer.train(epochs=100)

#Run the reservoir for the testing inputs
reservoir.run_given_inputs(X_test, calc_standardize=False)

#Predict the testing inputs with the ridge regression model
ridge_predictions = ridge_model.predict(reservoir.states_stand)

#Calculate ridge regression error
loss = trainer.MSELoss(ridge_predictions, Y_test)
print(f'Ridge Regression MSE for testing: {loss.item()}')

#Calculate attention model error
trainer.evaluate(reservoir.states_stand, Y_test)

N=250
for i in range(N):
    reservoir.one_step(X_test[i], standardize=True)

input = X_test[N]
prediction = []
for i in range(N):
    reservoir.one_step(input, standardize=True)
    pred_test = trainer.predict(reservoir.state_stand.reshape(1, -1)).T.squeeze()
    prediction.append(pred_test)
    input = pred_test

plt.plot(np.array(prediction)[:,0])
plt.plot(Y_test[N:,0])
plt.show()


#Get the next step prediction for the testing set
pred_test = trainer.predict(reservoir.states_stand)



#We can save and load the trained model under a specified path
#trainer.save_model('path_to_model.pt')
#trainer.load_model('path_to_model.pt')
