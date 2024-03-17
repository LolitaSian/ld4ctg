import torch
bart_latent = torch.tensor([0.0000, 0.1826, 0.3651, 0.5477, 0.7302], dtype=torch.float32)

print(bart_latent)

import geoopt.geoopt.manifolds.poincare.math as pm

latent = pm.expmap0(bart_latent)

print(latent)

reconstructed_latent = pm.logmap0(latent)

# 将切空间的原点映射回欧式空间
euclidean_point = pm.project(reconstructed_latent)
print(euclidean_point)