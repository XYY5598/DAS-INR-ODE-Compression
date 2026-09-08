import argparse
import os
import time

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Fitting GCI DAS data")
parser.add_argument("-f", "--file", required=True)
parser.add_argument("-c", "--starting_channel", type=int, required=True)
parser.add_argument(
    "-g",
    "--gpu",
    type=int,
    default=-1,
    help="GPU id; use -1 for CPU-only (no NVIDIA GPU).",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Training epochs (default: 8 on CUDA, 2 on CPU).",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=4096,
    help="DataLoader batch size (smaller helps on CPU memory/speed tradeoff).",
)
parser.add_argument(
    "--width-channels",
    type=int,
    default=1000,
    help="Number of channels to read from HDF5 (slice [c:c+width]). Default 1000 matches original script.",
)

args = parser.parse_args()

filename = args.file
basename = os.path.splitext(os.path.basename(filename))[0]
parts = basename.split("_")
# Original naming: decimator3_2023-07-16_*_UTC -> use 5th token; else use full stem.
filetime = parts[4] if len(parts) > 4 else basename
channel = args.starting_channel
width_ch = args.width_channels
gpu_id = args.gpu

if gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import h5py
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

if gpu_id >= 0 and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested (-g >= 0) but torch.cuda.is_available() is False.")


class DASDatasetOnTheFly(torch.utils.data.Dataset):
    def __init__(self, B, T, X, outputs):
        self.B = torch.Tensor(B)
        self.T = torch.Tensor(T)
        self.X = torch.Tensor(X)
        self.outputs = torch.Tensor(outputs)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        _t = self.T[index]
        _x = self.X[index]
        _Bv = torch.matmul(torch.Tensor(torch.concat([_t, _x], axis=-1)), self.B.T)
        X = torch.cat(
            [torch.cos(2 * torch.pi * _Bv), torch.sin(2 * torch.pi * _Bv)], axis=-1
        )
        y = self.outputs[index, :]
        return X, y


class RandomFourierFeature(torch.nn.Module):
    def __init__(self, nfeature, n_layers, n_outputs=1):
        super().__init__()
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.inputs = torch.nn.Linear(2 * nfeature, nfeature)

        for i in range(self.n_layers):
            setattr(self, f"ln{i+1}", torch.nn.Linear(nfeature, nfeature))

        self.outputs = torch.nn.Linear(nfeature, self.n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.relu

    def forward(self, x):
        x = self.relu(self.inputs(x))
        for i in range(self.n_layers):
            layer = getattr(self, f"ln{i+1}")
            x = self.relu(layer(x))
        x = 2 * self.sigmoid(self.outputs(x)) - 1
        return x


device = torch.device("cuda" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")
nepoch = (
    args.epochs
    if args.epochs is not None
    else (8 if device.type == "cuda" else 2)
)
batch_size = args.batch_size
ch_end = channel + width_ch
print(
    f"====REPORT: working on {filetime}, channel {channel}-{ch_end}, device={device} (gpu_id={gpu_id})===="
)
print(f"====REPORT: epochs={nepoch}, batch_size={batch_size}====")

df = h5py.File(filename, "r")
data = df["/Acquisition/Raw[0]/RawData"][:, channel:ch_end].T
df.close()

nc, ns = data.shape
print(f"got {nc} channel and {ns} sample: {data.shape}")

data -= np.mean(data, axis=-1, keepdims=True)
data /= np.max(np.abs(data), axis=-1, keepdims=True)
data = torch.Tensor(data)

in_x, in_t = torch.meshgrid(torch.arange(nc), torch.arange(ns))
T = in_t / (ns - 1)
X = in_x / (nc - 1)

n_feature = 196

B = torch.Tensor(np.random.normal(scale=20, size=(n_feature, 2)))

datasetonthefly = DASDatasetOnTheFly(
    B.to(device),
    T.reshape([-1, 1]).to(device),
    X.reshape([-1, 1]).to(device),
    data.reshape([-1, 1]).to(device),
)

data_loader = DataLoader(
    datasetonthefly, batch_size=batch_size, shuffle=True, num_workers=0
)

model = RandomFourierFeature(n_feature, n_layers=3)
for i in model.modules():
    if isinstance(i, torch.nn.Linear):
        i.weight.data.normal_(mean=0.0, std=0.1)
        i.bias.data.normal_(mean=0.0, std=0.1)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = MSELoss()
train_loss_log = []
t0 = time.time()
for t in range(nepoch):
    model.train()
    train_loss = 0

    for batch_id, batch in enumerate(data_loader):
        xb = batch[0].to(device)
        yb = batch[1].to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(data_loader)

    train_loss_log.append(train_loss)
    print(f"{t}: train loss: %.6f" % train_loss)
    if train_loss < 1e-4:
        break
print(time.time() - t0)

os.makedirs("./weights", exist_ok=True)
torch.save(
    model.state_dict(), f"./weights/rff_{filetime}_{channel}-{ch_end}.pt"
)
