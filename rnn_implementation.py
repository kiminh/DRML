import pickle
import torch
from torch.nn import functional as F
from random import seed
from random import randint
import torch.optim as optim
import torch.nn as nn
import numpy as np

path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
active_user_dict = pickle.load(open("{}/active_user_dict.pkl".format(path), "rb"))
active_label_dict = pickle.load(open("{}/active_user_label_dict.pkl".format(path), "rb"))
movie_dict = pickle.load(open("{}/movie_dict.pkl".format(path), "rb"))

#RMSE loss
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
criterion = RMSELoss()

# #rnn
class rnn_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn_model, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden=self.sigmoid(hidden)
        return hidden


input_size=96
hidden_size=96
output_size=96
rnn_mod=rnn_model(input_size,hidden_size,output_size)

cnt = 1
user_dynamics={}
for period in range(1,17):
    cn = 1
    periodic_dynamics = {}
    hidden = torch.tensor([np.ones(hidden_size).tolist()]).float()
    q_u = torch.eye(hidden_size, hidden_size).float()
    r_u = torch.eye(input_size, input_size).float()

    for user, item, labels in zip(active_user_dict.keys(), active_user_dict.values(), active_label_dict.values()):
        print('for user {}'.format(cnt))

        test_indx = []
        # random train test split (5 interactions are taken for the test from each period)
        seed(1)
        for _ in range(0, 5):
            indx = randint(0, len(item[period]) - 1)
            test_indx.append(indx)
        indexes = [i for i in range(0, len(item[period]) - 1)]
        train_indx = list(set(indexes) - set(test_indx))
        train_movie = [item[period][m] for m in train_indx]
        test_movie = [item[period][m] for m in test_indx]
        train_label = [active_label_dict[user][period][m] for m in train_indx]
        test_label = [active_label_dict[user][period][m] for m in test_indx]

        train_tensor = []
        for mov in train_movie:
            movie_info = movie_dict[mov]
            train_tensor.append(movie_info)

        train_tensor = torch.stack(train_tensor).float()

        test_tensor = []
        for mov in test_movie:
            movie_info = movie_dict[mov]
            test_tensor.append(movie_info)

        test_tensor = torch.stack(test_tensor).float()

        train_label = torch.unsqueeze(torch.tensor(train_label).float(), 1)
        test_label = torch.unsqueeze(torch.tensor(test_label).float(), 1)
        prev_loss = 9999
        hidden_bakup = hidden  # 6013 with director
        input = torch.mean(train_tensor, 0)
        while True:
            input = torch.matmul(input, r_u)
            hidden = torch.matmul(hidden, q_u)
            hidden_res = rnn_mod(input, hidden)
            y_pred = torch.matmul(train_tensor, hidden_res.t())
            loss = criterion(y_pred.view(-1, 1), train_label) + 0.01 * (torch.mean(q_u) ** 2 + torch.mean(r_u) ** 2)
            y_diff = y_pred.view(-1, 1) - train_label
            q_u = q_u - 0.001 * (torch.mean(hidden) * torch.mean(y_diff) + 0.01 * 2 * torch.mean(q_u))
            r_u = r_u - 0.001 * (torch.mean(input) * torch.mean(y_diff) + 0.01 * 2 * torch.mean(r_u))
            optimizer = optim.Adam(rnn_mod.parameters(), lr=0.001)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            print(loss)
            if prev_loss <= loss:
                break
            prev_loss = loss
            hidden_bakup = hidden_res
        periodic_dynamics.append(hidden_bakup)
        cn += 1
        hidden = hidden_bakup.view(-1,1).t()
        user_dynamics[user]=periodic_dynamics
        cnt += 1
pickle.dump(user_dynamics, open("{}/user_dynamics.pkl".format(path), "wb"))