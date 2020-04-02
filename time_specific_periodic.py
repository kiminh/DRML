import pickle
import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from random import seed
from random import randint


# RMSE loss
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


criterion = RMSELoss()


# plot function
def plot_function(x, y, title):
    plt.plot(x, y, 'r--')
    plt.yticks(np.arange(0, max(y) + 0.5, 0.5))
    plt.xticks(np.arange(0, max(x) + 1, 5))
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.title(title)
    plt.show()


# rnn
class rnn_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn_model, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.sigmoid(hidden)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        return output, hidden


class simple_meta_learning(torch.nn.Module):
    def __init__(self):
        super(simple_meta_learning, self).__init__()
        self.use_cuda = 'True'
        self.model = simple_neural_network(96)
        self.local_lr = 1e-4
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        for idx in range(num_local_update):
            user_rep = self.model(support_set_x)
            user_rep = torch.mean(user_rep, 0)
            support_set_y_pred = torch.matmul(support_set_x, user_rep.t())
            loss = criterion(support_set_y_pred.view(-1, 1), support_set_y)
            self.meta_optim.zero_grad()
            loss.backward()
            self.meta_optim.step()

        user_rep = self.model(query_set_x)
        user_rep = torch.mean(user_rep, 0)
        query_set_y_pred = torch.matmul(query_set_x, user_rep.t())
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        if self.use_cuda:
            support_set_xs = support_set_xs.cuda()
            support_set_ys = support_set_ys.cuda()
            query_set_xs = query_set_xs.cuda()
            query_set_ys = query_set_ys.cuda()
        query_set_y_pred = self.forward(support_set_xs, support_set_ys, query_set_xs, num_local_update)
        loss_q = criterion(query_set_y_pred.view(-1, 1), query_set_ys)
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        return loss_q


def dataset_prep(mov_list, movie_dict):
    data_tensor = []
    for mov in mov_list:
        movie_info = movie_dict[mov]
        data_tensor.append(movie_info.float())
    return torch.stack(data_tensor)


def training_function(ml_ss, support_set_x, support_set_y, query_set_x, query_set_y):
    training_loss = []

    for support_tensor, support_label, query_tensor, query_label in zip(support_set_x, support_set_y,
                                                                        query_set_x, query_set_y):
        user_loss = ml_ss.global_update(support_tensor, support_label, query_tensor, query_label, 1)
        training_loss.append(user_loss)
    loss = sum(training_loss) / len(training_loss)
    return loss


def testing_function(ml_ss, test_sup_set_x, test_sup_set_y, test_que_set_x, test_que_set_y):
    testing_loss = []

    for support_tensor, support_label, query_tensor, query_label in zip(test_sup_set_x, test_sup_set_y,
                                                                        test_que_set_x, test_que_set_y):
        user_loss = ml_ss.global_update(support_tensor, support_label, query_tensor, query_label, 5)
        testing_loss.append(user_loss)
    loss = sum(testing_loss) / len(testing_loss)
    return loss


def data_generation(active_user_dict, active_label_dict, movie_dict, period):
    user_data = {}
    for user, item, labels in zip(active_user_dict.keys(), active_user_dict.values(),
                                  active_label_dict.values()):
        query_indx = []
        temp_dict = {}
        # random support and query split
        seed(1)
        for _ in range(0, 5):
            indx = randint(0, len(item[period]) - 1)
            query_indx.append(indx)
        indexes = [i for i in range(0, len(item[period]) - 1)]
        support_indx = list(set(indexes) - set(query_indx))
        support_movie = [item[period][m] for m in support_indx]
        query_movie = [item[period][m] for m in query_indx]
        support_label = [active_label_dict[user][period][m] for m in support_indx]
        query_label = [active_label_dict[user][period][m] for m in query_indx]

        support_tensor = dataset_prep(support_movie, movie_dict)
        support_label = torch.unsqueeze(torch.tensor(support_label).float(), 1)
        query_label = torch.unsqueeze(torch.tensor(query_label).float(), 1)
        query_tensor = dataset_prep(query_movie, movie_dict)
        temp_dict[0] = support_tensor
        temp_dict[1] = support_label
        temp_dict[2] = query_tensor
        temp_dict[3] = query_label
        user_data[user] = temp_dict

    return user_data


# main fumction
if __name__ == "__main__":
    device = torch.device('cpu')
    path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
    active_user_dict = pickle.load(open("{}/final_user_dict.pkl".format(path), "rb"))
    active_label_dict = pickle.load(open("{}/final_label_dict.pkl".format(path), "rb"))
    movie_dict = pickle.load(open("{}/movie_dict.pkl".format(path), "rb"))


    user_dynamics = {}
    for period in range(1, 17):
        user_data = data_generation(active_user_dict, active_label_dict, movie_dict, period)
        ml_ss = simple_meta_learning()
        ml_ss.cuda()
        ml_ss.train()
        epoch = 1
        previous_loss = 999
        # Meta training
        training_loss_p = []
        x_tick = []
        if period <= 2:
            while epoch <= 50:
                training_loss = []
                for user in user_data.keys():
                    support_set_x = user_data[user][0]
                    support_set_y = user_data[user][1]
                    query_set_x = user_data[user][2]
                    query_set_y = user_data[user][3]
                    loss = training_function(ml_ss, support_set_x, support_set_y, query_set_x, query_set_y)
                    training_loss.append(loss)
                t_loss = sum(training_loss) / len(training_loss)
                print('Meta Training Loss for epoch {}= {}'.format(epoch, t_loss))
                if t_loss >= previous_loss:
                    break
                else:
                    previous_loss = t_loss
                    training_loss_p.append(t_loss)
                    x_tick.append(epoch)
                    epoch += 1

            # y = [l for l in training_loss_p]
            # y = torch.stack(y).to(device).detach().numpy()
            # plot_function(x_tick,y,'Meta Training Loss with the number of Local update is 1')
            # Meta Test
            testing_loss = []
            for user in user_data.keys():
                support_set_x = user_data[user][0]
                support_set_y = user_data[user][1]
                query_set_x = user_data[user][2]
                query_set_y = user_data[user][3]
                loss = testing_function(ml_ss, support_set_x, support_set_y, query_set_x, query_set_y)
                testing_loss.append(loss)
                print('Meta Test Loss for user {}= {}'.format(user, loss))
            t_loss = sum(testing_loss) / len(testing_loss)
            print('Meta Testing Loss = {}'.format(t_loss))

    print()
