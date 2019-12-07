import pt_pack as pt
from torch.utils.data import DataLoader

dataset = pt.GraphVqa2CpDataset(split='test', req_field_names=('q_labels', 'q_lens',))
loader = DataLoader(dataset, 2, False, num_workers=1)


for batch_idx, batch in enumerate(loader):
    print(batch)