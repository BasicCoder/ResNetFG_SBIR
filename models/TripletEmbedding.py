import torch as t
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from data import TripletDataLoader

from models.vgg import vgg16
from models.resnet import resnet34, resnet50, resnet101, resnet152
from utils.visualize import Visualizer
from torchnet.meter import AverageValueMeter


import tqdm
from utils.extractor import Extractor
from sklearn.neighbors import NearestNeighbors
import numpy as np
import shutil

class Config(object):
    def __init__(self):
        return

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class TripletNet(object):
    def __init__(self, opt):
        self.opt = opt
        # train config
        self.photo_root = opt.photo_root
        self.sketch_root = opt.sketch_root
        self.batch_size = opt.batch_size
        # self.device = opt.device
        self.epochs = opt.epochs
        self.lr = opt.lr

        self.photo_test = opt.photo_test
        self.sketch_test = opt.sketch_test
        self.save_dir = opt.save_dir

        self.logger = opt.logger
        # vis
        self.vis = opt.vis
        self.env = opt.env

        # fine_tune:
        self.fine_tune = opt.fine_tune
        self.model_root = opt.model_root


        # dataloader config
        data_opt = Config()
        data_opt.photo_root = opt.photo_root
        data_opt.sketch_root = opt.sketch_root
        data_opt.batch_size = opt.batch_size

        self.dataloader_opt = data_opt

        # triplet config
        self.margin = opt.margin
        self.p = opt.p

        # feature extractor net
        self.net = opt.net
        self.cat = opt.cat

    def _get_vgg16(self, pretrained=True):
        model = vgg16(pretrained)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)

        return model

    def _get_resnet34(self, pretrained=True):
        model = resnet34(pretrained=pretrained)
        model.fc = nn.Linear(in_features=512, out_features=125)

        return model

    def _get_resnet50(self, pretrained=True):
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=125)

        return model

    def _get_resnet101(self, pretrained=True):
        model = resnet101(pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=125)
        return model

    def _get_resnet152(self, pretrained=True):
        model = resnet152(pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=125)
        return model

    def train(self):
        self.triplet_loss_meter = AverageValueMeter()
        self.sketch_cat_loss_meter = AverageValueMeter()
        self.photo_cat_loss_meter = AverageValueMeter()

        self.photo_net.train()
        self.sketch_net.train()
        for ii, data in enumerate(self.dataset):

            self.photo_optimizer.zero_grad()
            self.sketch_optimizer.zero_grad()

            photo = data['P'].cuda()
            sketch = data['S'].cuda()
            label = data['L'].cuda()

            # vis.images(photo.detach().cpu().numpy()*0.5 + 0.5, win='photo')
            # vis.images(sketch.detach().cpu().numpy()*0.5 + 0.5, win='sketch')

            p_cat, p_feature = self.photo_net(photo)
            s_cat, s_feature = self.sketch_net(sketch)
            # category loss
            p_cat_loss = self.photo_cat_loss(p_cat, label)
            s_cat_loss = self.sketch_cat_loss(s_cat, label)

            self.photo_cat_loss_meter.add(p_cat_loss.item())
            self.sketch_cat_loss_meter.add(s_cat_loss.item())

            # p_cat_loss.backward(retain_graph=True)
            # s_cat_loss.backward(retain_graph=True)

            # triplet loss
            loss = (p_cat_loss + s_cat_loss) * self.opt.weight_cat

            for i in range(photo.size(0)):
                # negative
                negative_feature = t.cat([p_feature[0:i, :], p_feature[i + 1:, :]], dim=0)
                # print('negative_feature.size :', negative_feature.size())
                # photo_feature
                anchor_feature = s_feature[i, :]
                anchor_feature = anchor_feature.expand_as(negative_feature)
                # print('anchor_feature.size :', anchor_feature.size())

                # positive
                positive_feature = p_feature[i, :]
                positive_feature = positive_feature.expand_as(negative_feature)
                # print('positive_feature.size :', positive_feature.size())

                tri_loss = self.triplet_loss(anchor_feature, positive_feature, negative_feature)
                # print('tri_loss :', tri_loss)
                loss = loss + tri_loss
            # print('loss :', loss)
            # loss = loss / opt.batch_size

            loss.backward()

            self.photo_optimizer.step()
            self.sketch_optimizer.step()

            self.triplet_loss_meter.add(loss.item())
        if self.vis:
            self.visualizer.plot('triplet_loss', np.array([self.triplet_loss_meter.value()[0], self.photo_cat_loss_meter.value()[0],
                                           self.sketch_cat_loss_meter.value()[0]]),
                                 legend=['triplet_loss', 'photo_cat_loss', 'sketch_cat_loss'])

        self.triplet_loss_meter.reset()
        self.photo_cat_loss_meter.reset()
        self.sketch_cat_loss_meter.reset()

    def test(self):
        with t.no_grad():
            # extract photo feature
            extractor = Extractor(model=self.photo_net, vis=False)
            photo_data = extractor.extract_new(self.photo_test, batch_size=self.opt.test_batch_size)

            # extract sketch feature
            extractor.reload_model(self.sketch_net)
            sketch_data = extractor.extract_new(self.sketch_test, batch_size=self.opt.test_batch_size)

            photo_name = photo_data['name']
            photo_feature = photo_data['feature']

            sketch_name = sketch_data['name']
            sketch_feature = sketch_data['feature']

            nbrs = NearestNeighbors(n_neighbors=np.size(photo_feature, 0),
                                    algorithm='brute', metric='euclidean').fit(photo_feature)

            count_1 = 0
            count_5 = 0
            K = 5
            for ii,(query_sketch, query_name) in enumerate(zip(sketch_feature, sketch_name)):
                query_sketch = np.reshape(query_sketch, [1, np.shape(query_sketch)[0]])

                query_split = query_name.split('/')
                query_class = query_split[0]
                query_img = query_split[1]

                distance, indices = nbrs.kneighbors(query_sketch)
                # top K
                for i, indice in enumerate(indices[0][:K]):
                    retrievaled_name = photo_name[indice]
                    retrievaled_class = retrievaled_name.split('/')[0]

                    retrievaled_name = retrievaled_name.split('/')[1]
                    retrievaled_name = retrievaled_name.split('.')[0]
                    if retrievaled_class == query_class:
                        if query_img.find(retrievaled_name) != -1:
                            if i == 0:
                                count_1 += 1
                            count_5 += 1
                            break

            recall_1 = count_1 / (ii + 1)
            recall_5 = count_5 / (ii + 1)
        return recall_1, recall_5

    def save_checkpoint(self, is_best, only_save_best, epoch):
        photo_net_save_name = self.save_dir + '/photo' + '/photo_' + self.net + '_%s.pth' % epoch
        sketch_net_save_name = self.save_dir + '/sketch' + '/sketch_' + self.net + '_%s.pth' % epoch
        if not only_save_best:
            t.save(self.photo_net.state_dict(), photo_net_save_name)
            t.save(self.sketch_net.state_dict(), sketch_net_save_name)

        if is_best:
            if only_save_best:
                photo_net_save_name = self.save_dir + '/photo' + '/photo_' + self.net + '_best.pth'
                sketch_net_save_name = self.save_dir + '/sketch' + '/sketch_' + self.net + '_best.pth'
                t.save(self.photo_net.state_dict(), photo_net_save_name)
                t.save(self.sketch_net.state_dict(), sketch_net_save_name)
            else:
                shutil.copyfile(photo_net_save_name, self.save_dir + '/photo' + '/photo_' + self.net + '_best.pth')
                shutil.copyfile(sketch_net_save_name, self.save_dir + '/sketch' + '/sketch_' + self.net + '_best.pth')

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.8 ** (epoch // 1000))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)
    def update_test_frequency(self, epoch):
        if epoch > self.opt.niter + self.opt.niter_decay / 3:
            self.opt.test_freq = 5

    def run(self):
        if self.net == 'vgg16':
            self.photo_net = self._get_vgg16().cuda(self.opt.gpu_ids[0])
            self.sketch_net = self._get_vgg16().cuda(self.opt.gpu_ids[0])
        elif self.net == 'resnet34':
            self.photo_net = self._get_resnet34().cuda()
            self.sketch_net = self._get_resnet34().cuda()
        elif self.net == 'resnet50':
            self.photo_net = self._get_resnet50().cuda(self.opt.gpu_ids[0])
            self.sketch_net = self._get_resnet50().cuda(self.opt.gpu_ids[0])
        elif self.net == 'resnet101':
            self.photo_net = self._get_resnet101().cuda(self.opt.gpu_ids[0])
            self.sketch_net = self._get_resnet101().cuda(self.opt.gpu_ids[0])
        elif self.net == 'resnet152':
            self.photo_net = self._get_resnet152().cuda(self.opt.gpu_ids[0])
            self.sketch_net = self._get_resnet152().cuda(self.opt.gpu_ids[0])

        if len(self.opt.gpu_ids) > 0:
            assert (t.cuda.is_available())
            self.photo_net = t.nn.DataParallel(self.photo_net, self.opt.gpu_ids)
            self.sketch_net = t.nn.DataParallel(self.sketch_net, self.opt.gpu_ids)

        if self.fine_tune:
            photo_net_root = self.model_root
            sketch_net_root = self.model_root.replace('photo', 'sketch')

            self.photo_net.load_state_dict(t.load(photo_net_root, map_location=t.device('cpu')))
            self.sketch_net.load_state_dict(t.load(sketch_net_root, map_location=t.device('cpu')))


        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=self.p).cuda()
        self.photo_cat_loss = nn.CrossEntropyLoss().cuda()
        self.sketch_cat_loss = nn.CrossEntropyLoss().cuda()

        # optimizer
        self.schedulers = []
        self.optimizers = []
        self.photo_optimizer = t.optim.Adam(self.photo_net.parameters(), lr=self.lr, weight_decay=self.opt.weight_decay)
        self.sketch_optimizer = t.optim.Adam(self.sketch_net.parameters(), lr=self.lr, weight_decay=self.opt.weight_decay)
        self.optimizers.append(self.photo_optimizer)
        self.optimizers.append(self.sketch_optimizer)
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.opt))

        if self.vis:
            self.visualizer = Visualizer(self.opt)

        data_loader = TripletDataLoader(self.dataloader_opt)
        self.dataset = data_loader.load_data()

        self.best_recall_1 = 0
        for epoch in tqdm.tqdm(range(self.opt.epoch_count, self.opt.niter + self.opt.niter_decay + 1)):

            # adjust learning rate
            # self.adjust_learning_rate(self.photo_optimizer, epoch)
            # self.adjust_learning_rate(self.sketch_optimizer, epoch)

            if epoch % self.opt.test_freq == 0:
                # test
                recall_1, recall_5 = self.test()
                print('epoch:', epoch, '\trecall@1:', recall_1, '\trecall@5:', recall_5)
                self.logger.info('epoch:%d \trecall@1:%.15f \trecall@5:%.15f' % (epoch,  recall_1, recall_5))
                if self.vis:
                    self.visualizer.plot('recall', np.array([recall_1, recall_5]), legend=['recall@1', 'recall@5'])

                # remember best acc and save checkpoint
                is_best = recall_1 > self.best_recall_1
                self.best_recall_1 = max(recall_1, self.best_recall_1)
                # save model
                self.save_checkpoint(is_best, self.opt.only_save_best, epoch)

            # train
            self.train()
            # adjust learning rate
            self.update_learning_rate()
            # adjust test frequency
            self.update_test_frequency(epoch)
