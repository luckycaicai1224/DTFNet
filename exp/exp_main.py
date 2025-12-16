from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DTFNet
from tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from metrics import metric

from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import numpy as np
import mlflow

import torch
import torch.nn as nn

from einops import rearrange

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        cfg = vars(args)
        self.use_mlflow = self.args.use_mlflow
        if self.use_mlflow:
            name = self.args.model_id
            project = self.args.mlflow_project
            experiment = mlflow.set_experiment(project)
            mlflow.start_run(run_name=name)
            mlflow.log_params(cfg)

    def _build_model(self):
        model_dict = {
            'DTFNet': DTFNet,
        }
        model = model_dict[self.args.model].Model(self.args).float().cuda()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def check_param_in_optimizer(self,model_optim, param1):
        if model_optim is None:
            return False
        # 检查优化器中是否有这个参数
        for param_group in model_optim.param_groups:
            for param in param_group.keys():
                if param1 in str(param):
                    return True
        return False


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        return model_optim

    def _select_criterion(self):
        criterion = EnhancedHuberLoss(initial_delta=self.args.initTLV)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # outputs = None
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # if outputs is not None:
                #     batch_x = (batch_x + outputs) / 2

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DTFNet' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DTFNet' in self.args.model:
                        outputs = self.model(batch_x)
                        if isinstance(outputs, (tuple, list)):
                            outputs, draw_list, attn_list = outputs
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                        if isinstance(outputs, (tuple, list)):
                            outputs, draw_list, attn_list = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0

                if 'TST' in self.args.model:
                    outputs = outputs[:, :, f_dim:]
                else:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        if 'timm' in self.args.lradj:
            if 'cos' in self.args.lradj:
                scheduler = CosineLRScheduler(optimizer = model_optim,
                                              t_initial=train_steps - self.args.warmup_steps,
                                              lr_min=1e-8,
                                              warmup_t=self.args.warmup_steps,
                                              warmup_prefix=True,
                                              warmup_lr_init=1e-8
                                              )
            elif 'plateau' in self.args.lradj:
                scheduler = PlateauLRScheduler(optimizer=model_optim,
                                               decay_rate=self.args.decay_rate,
                                               patience_t=self.args.lr_patience,
                                               warmup_t=self.args.warmup_steps,
                                               warmup_lr_init=1e-8,
                                               lr_min=self.args.lr_min,
                                               mode='min'
                                        )

        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = self.args.pct_start,
                                                epochs = self.args.train_epochs,
                                                max_lr = self.args.learning_rate)
        
        # outputs = None
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # if 'XGBoost' in self.args.model:
                        #     outputs = self.model(batch_x)
                        if 'DTFNet' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'DTFNet' in self.args.model:
                        outputs = self.model(batch_x)
                        if isinstance(outputs, (tuple, list)):
                            outputs, draw_list, attn_list = outputs
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                            if isinstance(outputs, (tuple, list)):
                                outputs, draw_list, attn_list = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    outputs.requires_grad_(True)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.requires_grad_(True)

                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            if self.use_mlflow:
                log_dict = {'train/loss': train_loss, 'vali/loss': vali_loss, 'test/loss': test_loss}
                mlflow.log_metrics(log_dict, step=epoch)
                lr = model_optim.param_groups[0]['lr']
                mlflow.log_metric('lr', lr, step=epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if ('timm' in self.args.lradj and 'plateau' in self.args.lradj \
                    and model_optim.param_groups[0]['lr'] > self.args.lr_min) \
                    or epoch + 1 < self.args.warmup_steps:
                early_stopping.counter=0
                print("Reset Early stopping before Plateau Scheduler reach the lr_min")

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, metric=vali_loss)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        # visual always true in test function:
        self.args.visual = True

        if self.args.visual:
            self.model.visual = True  #self.model.model.visual

        with torch.no_grad():
            # outputs = None
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DTFNet' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DTFNet' in self.args.model:
                        outputs = self.model(batch_x)
                        if isinstance(outputs, (tuple, list)):
                            outputs, draw_list, attn_list = outputs
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                            if isinstance(outputs, (tuple, list)):
                                outputs, draw_list, attn_list = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = rearrange(preds, 'b l d -> b d l')
        trues = rearrange(trues, 'b l d -> b d l')
        inputx = rearrange(inputx, 'b l d -> b d l')


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        if self.use_mlflow:
            log_dict= {'best/test_mae': mae, 'best/test_mse': mse, "best/test_rse": rse}
            mlflow.log_metrics(log_dict)
            mlflow.end_run()


        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DTFNet' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DTFNet' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        # self.model.model.visual = True

        return

class EnhancedHuberLoss(nn.Module):
    def __init__(self, initial_delta=1.0, delta_lr=0.001, min_delta=0.1, max_delta=5.0):
        super().__init__()
        # 可学习的delta参数
        self.log_delta = nn.Parameter(torch.log(torch.tensor(initial_delta)))
        self.delta_lr = delta_lr
        self.min_delta = min_delta
        self.max_delta = max_delta

        # 用于delta更新的移动平均变量
        self.register_buffer('ema_error', torch.tensor(initial_delta))
        self.ema_alpha = 0.9

    @property
    def delta(self):
        # 返回受约束的delta值
        return torch.clamp(self.log_delta.exp(), self.min_delta, self.max_delta)

    def forward(self, input, target):
        delta = self.delta
        error = input - target
        abs_error = error.abs()

        # 计算基础Huber损失
        quadratic = torch.min(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + delta * linear

        # 动态更新delta参数
        if self.training:
            with torch.no_grad():
                # 更新EMA误差
                batch_mean_error = abs_error.mean()
                self.ema_error = self.ema_alpha * self.ema_error + (1 - self.ema_alpha) * batch_mean_error

                # 基于误差分布调整delta
                target_coverage = 0.8  # 希望80%的点在delta范围内
                delta_update = torch.sigmoid(self.ema_error - delta) - target_coverage
                self.log_delta.data += self.delta_lr * delta_update.to(self.log_delta.device)

        # 保存用于梯度调制的变量
        self.saved_for_backward = (error, delta)

        return loss.mean()

    def backward(self, grad_output):
        # 手动实现反向传播以应用梯度调制
        error, delta = self.saved_for_backward
        abs_error = error.abs()

        # 计算调制梯度
        modulated_grad = torch.where(
            abs_error <= delta,
            error,  # 原始梯度
            delta * torch.sin(error / delta)  # 调制梯度
        )

        # 应用调制后的梯度
        grad_input = grad_output * modulated_grad / (abs_error + 1e-8)
        return grad_input