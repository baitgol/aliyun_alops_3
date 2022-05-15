import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from feature.gen_features import features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import  logger
from model.utils import macro_f1

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class MLPModel(nn.Module):
    def __init__(self, num_inputs, num_classes=4):
        super(MLPModel, self).__init__()

        self.liner1 = nn.Sequential(nn.Linear(num_inputs, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU()
                                   )

        self.liner2 = nn.Sequential(nn.Linear(256, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU()
                                    )

        self.liner3 = nn.Sequential(nn.Linear(256, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU()
                                    )
        self.output = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.liner1(x)
        x = self.liner2(x)
        x = self.liner3(x)
        x = self.output(x)
        return x

class MLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 多少step保存模型
def train(train_data_loader, eval_data_loader, model, criterion, optimizer, num_epoch, log_step_interval, save_step_interval, eval_step_interval,
          save_path, resume=""):

    start_epoch = 0
    start_step = 0

    macro_f1_scores = []
    if resume != "":
        # 加载之前训练过的模型
        logger.warning(f"loading from {resume}")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint["step"]

    for epoch_index in range(start_epoch, num_epoch):
        print('Epoch {}/{}'.format(epoch_index, num_epoch-1))
        print('-' * 10)
        ema_loss = 0
        num_batches = len(train_data_loader)
        for batch_index, (X, y) in enumerate(train_data_loader):
            optimizer.zero_grad()
            step = epoch_index*num_batches+batch_index+1

            output = model(X)
            cross_entropy_loss = criterion(output, y)
            ema_loss = 0.9 * ema_loss + 0.1 * cross_entropy_loss
            cross_entropy_loss.backward()
            optimizer.step()
            # logger.warning(f"第{epoch_index}个epoch的学习率：{optimizer.param_groups[0]['lr']}")

            if step % log_step_interval == 0:
                logger.warning(f"epoch_index: {epoch_index}, batch_index: {batch_index}, ema_loss: {ema_loss}")

            if step % save_step_interval == 0:
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, f"step_{step}.pt")
                torch.save({"epoch": epoch_index,
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": cross_entropy_loss
                            }, save_file)
                logger.warning(f"checkpoint has been saved in {save_file}")

            if step % eval_step_interval == 0:
                logger.warning("start to do evaluation....")
                model.eval()
                eval_ema_loss = 0
                total_acc_count = 0
                total_count = 0
                eval_output_lst = []
                eval_true_lst = []
                for eval_batch_index, (eval_X, eval_y) in enumerate(eval_data_loader):
                    total_count += eval_X.shape[0]
                    eval_output = model(eval_X)
                    #total_acc_count += (torch.argmax(eval_output, dim=-1) == eval_y).sum().item()
                    eval_cross_entropy_loss = criterion(eval_output, eval_y)
                    eval_ema_loss = 0.9 * eval_ema_loss + 0.1 * eval_cross_entropy_loss

                    eval_output_lst.extend(eval_output.argmax(1).detach().numpy())
                    eval_true_lst.extend(eval_y.detach().numpy())

                macro_f1_scores.append(macro_f1(eval_true_lst, eval_output_lst))
                logger.warning(f"eval_ema_loss: {eval_ema_loss}, eval_macro_f1: {macro_f1(eval_true_lst, eval_output_lst)}")
                model.train()

    print(macro_f1_scores)



def nn_train(X_train, y_train):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, shuffle=True, stratify=y_train, random_state=2022)
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr.values)
    X_val = scaler.transform(X_val.values)

    # X_tr = torch.from_numpy(X_tr)
    # X_val = torch.from_numpy(X_val)
    X_tr = torch.tensor(X_tr, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_tr = torch.tensor(y_tr.values, dtype=torch.int64)
    y_val = torch.tensor(y_val.values, dtype=torch.int64)

    num_feature = X_train.shape[1]
    model = MLPModel(num_feature)

    print("模型总参数:", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    train_dataset = MLPDataset(X_tr, y_tr)
    eval_dataset = MLPDataset(X_val, y_val)

    train_ds = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_ds = DataLoader(eval_dataset, batch_size=64)

    resume = ""
    train(train_ds, eval_ds, model, criterion, optimizer, num_epoch=100, log_step_interval=100,
          save_step_interval=500, eval_step_interval=300, save_path="model/nn_baseline", resume=resume)