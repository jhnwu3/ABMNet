# problem with the code is that the spatial data can only be run on the cluster, because it's so massive, which means we can't run anything locally
import torch 
import gc
from torch.cuda.amp import autocast
from modules.utils.graph import *
from modules.data.spatial import *
from modules.models.spatial import *
# Cytotoxic CD8+ T Cells, Cancer, Exhausted CD8+ T Cells, Dead Cancer Cells, Ignore, Ignore, TAMs, Ignore, Ignore
# 0,1,2,3,6
data = SingleInitialConditionDataset("../gdag_data/gdag_graph_data.pickle", channels=[0,1,2,3,6])
train_size = int(0.8 * len(data))
test_size = len(data) - train_size

# split dataset, into training set and test
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

nEpochs = 2
print("nEpochs:", nEpochs)
# for manual testing, load everything at once, and train
# model = GCN(n_features=input_graph.size()[1], n_classes=output_graphs_chunk[0].size()[1])
model = GCNComplex(n_features=data.n_inputs, n_classes= data.n_outputs, n_rates=data.n_rates, hidden_channels=32)
model.train()
model = model.double()
optimizer = torch.optim.AdamW(model.parameters())
criterion = torch.nn.MSELoss()

device = ""
scaler = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    using_gpu = True
    scaler = torch.cuda.amp.GradScaler()
else:
    device = torch.device("cpu")
    using_gpu = False
    
model = model.to(device)
criterion = criterion.to(device)

dataloader = torch.utils.data.DataLoader(train_data, batch_size=None, shuffle=True) 
for epoch in range(nEpochs):
    loss_per_epoch = 0
    for rates, output_graph in dataloader:
        optimizer.zero_grad()
        
        if using_gpu:
            loss = 0
            with autocast():
                out = model(data.initial_graph.to(device), data.edges.to(device), rates.to(device))
                loss = criterion(out, output_graph.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: 
            loss.backward()
            loss_per_epoch += loss.item()
            optimizer.step()
            
        loss_per_epoch += loss.cpu().item()
        if using_gpu:
            loss = None 
            
    if using_gpu and epoch  % 5 == 0:
        torch.cuda.empty_cache()
        gc.collect()
    
    if epoch % 1 == 0:
        print("mem:", torch.cuda.memory_allocated(), " cached:", torch.cuda.memory_cached())
        print("Epoch:", epoch, " Loss:", loss_per_epoch)   

print("DONE!") 
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=None, shuffle=True)
model.eval()
test_loss = 0
with torch.no_grad():
    for rates, output_graph in test_dataloader:
        out = model(data.initial_graph.to(device), data.edges.to(device), rates.to(device))
        test_loss += criterion(out.detach(), output_graph.to(device))

print("Test Average MSE:", test_loss / len(test_data))


# for i in range(len(output_graphs_chunk)):
#     print(output_graphs_chunk[i].size())
# train_profiled(input_graph, output_graphs_chunk, rates_chunk, edges)




# model = train_giuseppe_surrogate_pkl(loaded_data, nEpochs=20)


