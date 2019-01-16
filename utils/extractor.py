import os
from PIL import Image
import torch as t
import torchvision as tv
from torch import nn
import pickle
import numpy as np

class Extractor(object):
    def __init__(self, pretrained=False, vis=True, model=None):
        if model:
            self.model = model
        else:
            model = tv.models.resnet34(pretrained)
            del model.fc
            model.fc = lambda x: x
            model.cuda()

            self.model = model

        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.vis = vis

    # extract the inputs' feature via self.model
    # the model's output only contains the inputs' feature
    @t.no_grad()
    def extract(self, data_root, out_root=None):
        feature = []
        name = []

        self.model.eval()

        cnames = sorted(os.listdir(data_root))
        for cname in cnames:
            fnames = sorted(os.listdir(os.path.join(data_root, cname)))
            for fname in fnames:
                path = os.path.join(data_root, cname, fname)

                image = Image.open(path)
                image = self.transform(image)
                image = image[None]
                image = image.cuda()

                i_feature = self.model(image)

                feature.append(i_feature.cpu().sequeeze().numpy())
                name.append(cname + '/' + fname)
        # 'name': category_name/file_name 'feature' : (1, D) array
        data = {'name': name, 'feature': feature}
        if out_root:
            out = open(out_root, 'wb')
            pickle.dump(data, out)
            out.close()
        return data

    # extract the inputs' feature via self.model
    # the model's output contains both the inputs' feature and category info
    @t.no_grad()
    def extract_new(self, data_root, out_root=None, batch_size=1):
        feature = []
        name = []

        self.model.eval()

        cnames = sorted(os.listdir(data_root))
        for cname in cnames:
            fnames = sorted(os.listdir(os.path.join(data_root, cname)))
            for fname in fnames:
                name.append(cname + '/' + fname)

        fname_chunks = [name[i:i+batch_size] for i in range(0, len(name), batch_size)]

        for fname_chunk in fname_chunks:
            image_chunk = [self.transform(Image.open(os.path.join(data_root, fname))) for fname in fname_chunk]
            images = t.stack(image_chunk, 0)
            images = images.cuda()

            _, i_feature = self.model(images)
            if batch_size == 1:
                i_feature = [i_feature.cpu().squeeze().numpy()]
            else:
                i_feature = np.vsplit(i_feature.cpu().squeeze().numpy(), i_feature.size(0))
            i_feature = [np.squeeze(i) for i in i_feature]
            feature.extend(i_feature)

        data = {'name': name, 'feature': feature}
        if out_root:
            out = open(out_root, 'wb')
            pickle.dump(data, out)
            out.close()

        return data

    # reload model with model file
    # the reloaded model contains fully connection layer
    def reload_state_dict_with_fc(self, state_file):
        temp_model = tv.models.resnet34(pretrained=False)
        temp_model.fc = nn.Linear(512, 125)
        temp_model.load_state_dict(t.load(state_file))

        pretrained_dict = temp_model.state_dict()

        model_dict = self.model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # reload model with model file
        # the reloaded model doesn't contain fully connection layer

    def reload_state_dic(self, state_file):
        self.model.load_state_dict(t.load(state_file))

    # reload model with model object directly
    def reload_model(self, model):
        self.model = model