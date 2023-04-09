import matplotlib.pyplot as plt
import numpy as np
import json

exp_title = "best model(CNN)"

path_queue = [
    "/home/junyeob/AI502/exp/run_5/cnn-best-lr0005.json",
]
legends = [
    "lr0.005",
    "lr0.001",
    "lr0.0005",
]
for i, path_to_load in enumerate(path_queue):
    with open(path_to_load, 'r') as f:
        data = json.load(f)

    epochs = np.linspace(1, len(data['loss']), len(data['loss']))
    losses = np.array(data['loss'])

    plt.plot(epochs, losses)
    plt.title(f"{exp_title} - Loss")
    plt.xlabel("epochs")
    plt.ylabel("losses")
# plt.legend()
plt.savefig(f"{exp_title}_loss.png")    
plt.cla()

for i, path_to_load in enumerate(path_queue):
    with open(path_to_load, 'r') as f:
        data = json.load(f)

    epochs = np.linspace(1, len(data['loss']), len(data['loss']))
    accs = np.array(data['acc'])

    plt.plot(epochs, accs)
    plt.title(f"{exp_title} - Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accs(%)")
plt.legend()
plt.savefig(f"{exp_title}_acc.png")