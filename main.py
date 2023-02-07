import torch
from eigenpro3.kernels import laplacian
from eigenpro3 import KernelModel, Dataset

n, p, d, c = 1000, 100, 10, 3
bw = 1.

samples = torch.randn(n, d)
centers = torch.randn(p, d)
labels = torch.randn(n, c)

kernel_fn = lambda x, z: laplacian(x, z, bandwidth=1.)

data = Dataset(samples, kernel_fn=kernel_fn, precondition=True, top_q=10)

model = KernelModel(centers=centers, kernel_fn=kernel_fn)

model.fit(data, labels, batch_size=12)