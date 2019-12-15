import pt_pack as pt
from torch.utils.data import DataLoader
import torch

dataset = pt.GraphGqaDataset(gpu_num=1, use_filter=True)
loader = DataLoader(dataset, 60, True, num_workers=4, collate_fn=dataset.collate_fn)


for batch_idx, batch in enumerate(loader):
    for key, value in batch.items():
        if torch.is_tensor(value):
            pt.to_cuda(value, 'cuda')