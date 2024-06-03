## 3_make_prediction.py

# This is the last step of this method. After the model is trained, we can use this model to
# perform the analytic continuation. An example is given here.

import h5
import torch
import numpy as np
import matplotlib.pyplot as plt
from ACANN import ACANN

model = ACANN(20, 2001, [43, 93, 200, 431, 929], drop_p=0.09).double()

## Example input. This is a bosonic DLR
dlr = np.array([0.18787499515589712,
                0.172752067098271,
                0.14999679771385863,
                0.11895092360058762,
                0.09338140249557225,
                0.06559366886654071,
                0.04520263716228265,
                0.03768131335892698,
                0.03206579313879842,
                0.029300016701990755,
                0.028420024081208622,
                0.03134132911916321,
                0.03461639533063508,
                0.040861317930030984,
                0.0593892561121784,
                0.08851801911089098,
                0.11185006742604406,
                0.1441077196207387,
                0.17274267433434076,
                0.18788582942676993,
])

omega = np.linspace(-8, 8, 2001)

## Load the model, which should also be bosonic
# `map_location` can be designated if training and evaluating are not on the same device
model.load_state_dict(torch.load('b_20240603080817.pth', map_location=torch.device('cpu')))
model.eval()

new_data = torch.from_numpy(dlr).unsqueeze(0).double()
new_data = new_data.to(model.layers[0].weight.device)

with torch.no_grad():
    predictions = model(new_data)
    predicted_function = predictions.squeeze()  # Remove any extra dimensions

# Plot the function
plt.plot(omega, predicted_function, label='Analytic continuation')
plt.legend()
plt.show()
