# coding=utf8

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import sys
from data import MyDataset
import logging
from model.model import Model
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

torch.cuda.manual_seed(1)

logging.basicConfig(level=logging.INFO,
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


def train(epoch):
    start_time = time.time()
    model.train()
    loss_epoch = []
    for i, (lob, label) in enumerate(dataloader_train):
        optimizer.zero_grad()
        lob, label = lob.cuda(), label.cuda()
        pred = model(lob)
        pred = torch.squeeze(pred)
        loss = loss_fn(pred, label)
        loss_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logging.info("Epoch: %d/%s, Step: %d/%d,  Loss: %f", 
                             epoch , epochs, i, len(dataloader_train),
                             float(loss), )
    end_time = time.time()
    train_loss.append(np.mean(loss_epoch))
    logging.info("Train Epoch %d/%s Finished | Train Loss: %f",
                     epoch, epochs, train_loss[-1])
    return np.mean(train_loss)

@torch.no_grad()
def evaling(epoch):
    start_time = time.time()
    model.eval()
    loss_epoch = []
    for (lob, label) in tqdm(dataloader_test, ncols=40):
        lob, label = lob.cuda(), label.cuda()
        outputs = model(lob)
        outputs = torch.squeeze(outputs)
        loss = loss_fn(outputs, label)
        loss_epoch.append(loss.item())


    end_time = time.time()
    test_loss.append(np.mean(loss_epoch))
    curr_loss = test_loss[-1]
    logging.info('Evaluating Network.....')
    logging.info('Test set: Epoch: {}, Current Loss: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        curr_loss,
        end_time - start_time
    ))

    return curr_loss

def plot_curve(train_loss, test_loss):
    # x轴数据（epochs或迭代次数）
    epochs = range(1, len(train_loss) + 1)

    # 绘制训练损失曲线
    plt.plot(epochs, train_loss, 'b', label='Train Loss')

    # 绘制测试损失曲线
    plt.plot(epochs, test_loss, 'r', label='Test Loss')

    
    # 设置图例和标签
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 显示网格线
    plt.grid(True)

    # 显示图形
    plt.show()
    plt.savefig('loss_curve.png')


if __name__ == "__main__":
    '''data'''

    k = 0
    dep = 1
    bs = 256
    Xs = np.load('./data/X.npy')
    ys = np.load('./data/y.npy')
    dataset_train = MyDataset(Xs=Xs, ys=ys, split='train', T=50)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=bs, shuffle=True)
    dataset_test = MyDataset(Xs=Xs, ys=ys, split='test', T=50)
    
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=bs, shuffle=False)

    model_name = 'model' # ocet deeplob deepfolio 
    save_k = ['k_50', 'k_100']
    save_path = 'model_save/' + save_k[k] + '/' + model_name + '/'
    os.makedirs(save_path, exist_ok=True)

    '''model'''
    mode = model_name  

    model = Model()
    
    model = model.cuda()
  
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay= 0.001)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    logging.info('  Model = %s', str(model))
    logging.info('  Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('  Train num = %d', len(dataset_train))
    logging.info('  Test num = %d', len(dataset_test))

    epochs = 50
    train_loss = []

    best_epoch = 1
    best_loss = 2.0
    test_loss = []

    for epoch in range(1, epochs + 1):
        tra_loss = train(epoch)
        curr_loss = evaling(epoch)
        train_loss.append(tra_loss)
        test_loss.append(curr_loss)
        if curr_loss < best_loss:
            best_epoch = epoch
            best_loss = curr_loss
        output_path = save_path + mode + '_epoch_' + str(epoch) + '_valloss_' + str(format(curr_loss, '.4f')) + '.pth'
        torch.save(model.state_dict(), output_path)
        logging.info("The best epoch is : {}, loss is : {:.4f}".format(
            best_epoch,
            best_loss
        ))
    plot_curve(train_loss, test_loss)
