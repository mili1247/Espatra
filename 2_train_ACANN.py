## 2_train_ACANN.py

# This is the second step after training data are generated in the `Database/`. It uses a neural
# network to fit the inverse functional, whose input is the Green's function (parametrized by DLR) 
# and outputs the spectral function.

import h5
import torch
import datetime
from ACANN import *
from torch.nn.modules.loss import KLDivLoss,L1Loss, SmoothL1Loss
from torch.optim import Adam,Rprop,Adamax, RMSprop,SGD,LBFGS
from torch.utils.data import DataLoader

# Put all the data to be used as a list here
training_files = [
    "Database/training_20240528101533.h5",
    "Database/training_20240529041252.h5",
    "Database/training_20240529055051.h5",
]
validation_files = [
    "Database/validation_20240528101743.h5",
    "Database/validation_20240529040940.h5",
]

## Create the network
# The first argument should be equal to the number of DLR coefficients
# The second argument should be equal to the number of omega grid points
# The third argument is how the layers and nodes arranged for the neural network
model = ACANN(20, 2001, [43, 93, 200, 431, 929], drop_p=0.09).double()

print("Model created")


###################################################################################################

print(f"Using CUDA? {torch.cuda.is_available()}")
print("Starting ACANN")

# Import the data
training_data = load_data(training_files)
validation_data = load_data(validation_files)
trainloader = DataLoader(training_data, batch_size=2000, shuffle=True)
validationloader = DataLoader(validation_data, batch_size=1000)

print("Training and validation data loaded successfully.")

# Define a function for computing the validation score
def validation_score(nn_model):
    nn_model.eval()
    val_error = L1Loss()
    with torch.no_grad():
        G_val, A_val = next(iter(validationloader))
        prediction = nn_model.forward(G_val)
        score = val_error(prediction, A_val)
    nn_model.train()
    return score.item()


#Define the loss
error = L1Loss()
#Define the optimizer
optimizer = Adam(model.parameters())
#RMSPRO 10 - 2e-3
#ADAM 10 - 1.2e-3

# Training parameters
epochs = 1000
step = -1
print_every = 250
print("Starting the training")

# Training
for e in range(epochs):
    model.train()
    #  Load a minibatch
    for G,A in trainloader:
        step += 1
        # restart the optimizer
        optimizer.zero_grad()
        # compute the loss
        prediction = model.forward(G)
        loss = error(prediction, A)
        # Compute the gradient and optimize
        loss.backward()
        optimizer.step()

        # Write the result
        if step % print_every == 0:
            step = 0
            print("Epoch {}/{} : ".format(e + 1, epochs),
                  "Training MAE = {} -".format(loss.item()),
                  "Validation MAE = {}".format(validation_score(model)))

# Save the model
time_suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
torch.save(model.state_dict(),f'model_{time_suffix}.pth')

print(f"Model is written in `model_{time_suffix}.pth`")
