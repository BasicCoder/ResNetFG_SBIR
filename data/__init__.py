from data.triplet_input import TripletDataset
import torch.utils.data

class TripletDataLoader(object):
    def __init__(self, opt):
        self.dataset = TripletDataset(opt.photo_root, opt.sketch_root)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=opt.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data