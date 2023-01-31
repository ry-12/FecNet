import math
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model import RNN
from tqdm import tqdm
from recorder import *
from metrics import *
from dataloader import *
from utils import *
torch.backends.cudnn.benchmark=True

# partly cite from SimVP, sincere Thanks!
# https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction

class Run:
    def __init__(self, args):
        super(Run, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        set_seed(self.args.seed)
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        self._get_data()
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = RNN(args.num_hidden, args.img_channel, args.filter_size, args.stride, args.sr_size,
                            args.total_length, args.input_length, args.img_height, args.img_width).to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        args = self.args
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.T_max
        )
        return self.optimizer

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def schedule_sampler(self, n):
        args = self.args
        random_flip = np.random.random_sample((args.batch_size, args.total_length - args.input_length - 1))

        if n <= args.iter:
            p = math.cos(n * math.pi / 2 * args.iter)
        else:
            p = 0

        true_token = (random_flip < p)
        # print(true_token.shape)
        ones = np.ones((args.img_width, args.img_width, args.img_channel))
        zeros = np.zeros((args.img_width, args.img_width, args.img_channel))
        real_input_flag = []
        for i in range(args.batch_size):
            for j in range(args.total_length - args.input_length - 1):
                if true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
        real_input_flag = np.array(real_input_flag)
        real_input_flag = np.reshape(real_input_flag, (args.batch_size,
                                                       args.total_length - args.input_length - 1,
                                                       args.img_width,
                                                       args.img_width, args.img_channel))
        mask = torch.from_numpy(real_input_flag).to(torch.float32)
        return mask

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for i, (batch_x, batch_y) in enumerate(train_pbar):
                n = i + epoch * len(self.train_loader)
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                inputs = torch.cat([batch_x, batch_y], dim=1)
                mask = self.schedule_sampler(n).to(self.device)
                out, loss = self.model(inputs, mask)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            train_loss = np.average(train_loss)
            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader, epoch)
                    if epoch % (args.log_step * 1) == 0:
                        self._save(name=str(epoch))
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader, epoch):
        args = self.args
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            inputs = torch.cat([batch_x, batch_y], dim=1)
            mask = torch.zeros(args.val_batch_size, args.input_length - 1, args.img_width,
                               args.img_width, args.img_channel).to(self.device)

            out, loss = self.model(inputs, mask)
            pre = out[:, :, args.input_length - 1:, :, :].permute(0, 2, 1, 3, 4)

            # print(pre.shape)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                pre, batch_y], [preds_lst, trues_lst]))

            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            inputs = torch.cat([batch_x, batch_y], dim=1)
            mask = torch.zeros(args.batch_size, args.input_length - 1, args.img_width,
                               args.img_width, args.img_channel).to(self.device)

            out, loss = self.model(inputs, mask)
            pre = out[:, :, args.total_length - args.input_length - 1:, :, :].permute(0, 2, 1, 3, 4)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                batch_x, batch_y, pre], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path + '/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        return mse
