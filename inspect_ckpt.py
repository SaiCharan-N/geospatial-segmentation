import torch
ckpt = torch.load('models/road.pth', map_location='cpu')
print("Keys in checkpoint:")
items = ckpt.items() if isinstance(ckpt, dict) else ckpt.state_dict().items()
for k, v in items:
    print(f"{k}: {v.shape}")
