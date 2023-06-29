import torch
import matplotlib.pyplot as plt



# Load pre-trained weights (optional)
model = torch.load('model/l3p_100k_large_batch_normed.pt', map_location=torch.device('cpu'))

# Get the state dictionary
state_dict = model.state_dict()

# Extract the weights
weights = [] 
for name, param in state_dict.items():
    if 'weight' in name:
        weights.append(param)
        print(param.size())
# Visualize the weights
fig, axs = plt.subplots(len(weights), figsize=(30,30))
for i, weight in enumerate(weights):
    if len(weight.size()) > 1:
        axs[i].imshow(weight.detach().numpy())
        axs[i].set_title(f'Layer {i+1} Weights')
plt.savefig("Nithya.png")
plt.show()
