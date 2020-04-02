import pickle
from copy import deepcopy
from random import randint
from random import seed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.nn import functional as F


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
        combined = torch.cat((input,hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        return output


# simple nn
class simple_neural_network(torch.nn.Module):
    def __init__(self, input_dim):
        super(simple_neural_network, self).__init__()
        self.use_cuda = 'True'
        self.fc1 = nn.Linear(input_dim, 64)
        self.i2o = nn.Linear(64, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        hidden_out = self.fc1(input)
        hidden_out = F.relu(hidden_out)
        output = self.i2o(hidden_out)
        output = self.sigmoid(output)
        return output


class simple_meta_learning(torch.nn.Module):
    def __init__(self):
        super(simple_meta_learning, self).__init__()
        self.use_cuda = 'True'
        self.model = simple_neural_network(96)
        self.local_lr = 5e-4
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update,rnn_input):
        for idx in range(num_local_update):
            user_rep = self.model(support_set_x)
            user_rep = torch.mean(user_rep, 0)
            support_set_y_pred_1 = torch.matmul(support_set_x, user_rep.t())
            support_set_y_pred_2=torch.matmul(support_set_x, rnn_input.t())
            support_set_y_pred=support_set_y_pred_1+support_set_y_pred_2
            loss = criterion(support_set_y_pred.view(-1,1), support_set_y)
            self.meta_optim.zero_grad()
            loss.backward(retain_graph=True)
            self.meta_optim.step()

        user_rep = self.model(query_set_x)
        user_rep = torch.mean(user_rep, 0)
        query_set_y_pred_1 = torch.matmul(query_set_x, user_rep.t())
        query_set_y_pred_2 = torch.matmul(query_set_x, rnn_input.t())
        query_set_y_pred = query_set_y_pred_1 + query_set_y_pred_2
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred.view(-1,1),user_rep

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update,rnn_input):
        if self.use_cuda:
            support_set_xs = support_set_xs.cuda()
            support_set_ys = support_set_ys.cuda()
            query_set_xs = query_set_xs.cuda()
            query_set_ys = query_set_ys.cuda()
            rnn_input=rnn_input.cuda()
        query_set_y_pred,time_spec = self.forward(support_set_xs, support_set_ys, query_set_xs, num_local_update,rnn_input)
        loss_q = criterion(query_set_y_pred.view(-1, 1), query_set_ys)
        self.meta_optim.zero_grad()
        loss_q.backward(retain_graph=True)
        self.meta_optim.step()
        self.store_parameters()
        return loss_q,time_spec


def dataset_prep(mov_list, movie_dict):
    data_tensor = []
    for mov in mov_list:
        movie_info = movie_dict[mov]
        data_tensor.append(movie_info.float())
    return torch.stack(data_tensor)


def training_function(ml_ss, support_set_x, support_set_y, query_set_x, query_set_y,rnn_input):

    user_loss,time_spec = ml_ss.global_update(support_set_x, support_set_y, query_set_x, query_set_y, 1,rnn_input)

    return user_loss,time_spec


def testing_function(ml_ss, test_sup_set_x, test_sup_set_y, test_que_set_x, test_que_set_y,rnn_input):
    user_loss,time_spec = ml_ss.global_update(test_sup_set_x, test_sup_set_y, test_que_set_x, test_que_set_y, 5,rnn_input)
    return user_loss,time_spec


def data_generation(active_user_dict, active_label_dict, movie_dict, period):
    user_data={}
    for user, item, labels in zip(active_user_dict.keys(), active_user_dict.values(),
                                  active_label_dict.values()):
        query_indx = []
        temp_dict={}
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
        temp_dict[0]=support_tensor
        temp_dict[1]=support_label
        temp_dict[2]=query_tensor
        temp_dict[3]=query_label
        user_data[user]=temp_dict

    return user_data


# main fumction
if __name__ == "__main__":
    device = torch.device('cpu')
    path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
    active_user_dict = pickle.load(open("{}/final_user_dict.pkl".format(path), "rb"))
    active_label_dict = pickle.load(open("{}/final_label_dict.pkl".format(path), "rb"))
    movie_dict = pickle.load(open("{}/movie_dict.pkl".format(path), "rb"))

    input_size = 96
    hidden_size = 96
    output_size = 96
    rnn_mod = rnn_model(input_size, hidden_size, output_size)
    rnn_mod.cuda()
    user_dynamics={}
    time_specific_user={}
    intial_hidden=torch.zeros(hidden_size).float()
    for period in range(1, 17):
        user_data=data_generation(active_user_dict,active_label_dict,movie_dict,period)
        ml_ss = simple_meta_learning()
        ml_ss.cuda()
        ml_ss.train()
        epoch = 0
        previous_loss = 999
        # Meta training
        training_loss_p = []
        x_tick = []
        train_user=[]
        test_user=[]
        cnt=0
        for user in user_data.keys():
            if cnt<150:
                train_user.append(user)
            else:
                test_user.append(user)
            cnt+=1

        if period==1:
            for user in user_data.keys():
                user_dynamics[user]=torch.reshape(intial_hidden,(1,96))
        else:
            previous_loss=999
            while True:
                rnn_loss=[]
                for user in user_data.keys():
                    support_set_x = user_data[user][0]
                    support_set_y = user_data[user][1]
                    query_set_x = user_data[user][2]
                    query_set_y = user_data[user][3]
                    train_x = torch.cat((support_set_x, query_set_x),dim=0)
                    train_y = torch.cat((support_set_y, query_set_y),dim=0)
                    hidden=user_dynamics[user]
                    input=torch.mean(train_x,dim=0)
                    train_x=train_x.cuda()
                    train_y=train_y.cuda()
                    hidden=torch.reshape(hidden,(1,96)).cuda()
                    input = input.cuda()
                    hidden_res = rnn_mod(input, hidden)
                    hidden_res = hidden_res.cuda()
                    y_pred_1 = torch.matmul(train_x, hidden_res.t())
                    y_pred_2 = torch.matmul(train_x, time_specific_user[user].t())
                    y_pred = y_pred_1 + y_pred_2
                    loss = criterion(y_pred.view(-1, 1), train_y)
                    optimizer = optim.Adam(rnn_mod.parameters(), lr=1e-4)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    rnn_loss.append(loss)
                    hidden = hidden_res
                    user_dynamics[user] = hidden_res

                rn_loss=sum(rnn_loss)/len(rnn_loss)

                # print('RNN loss at epoch {}={}'.format(epoch, rn_loss))
                if epoch%25==0:
                    print('RNN loss at epoch {}={}'.format(epoch,rn_loss))
                epoch+=1
                if previous_loss <= rn_loss or epoch==300:
                    break
                previous_loss = rn_loss

        if period<=2:
            epoch=0
            previous_loss = 999
            while True:
                training_loss=[]
                for user in train_user:
                    support_set_x=user_data[user][0]
                    support_set_y=user_data[user][1]
                    query_set_x=user_data[user][2]
                    query_set_y=user_data[user][3]
                    loss,time_spec_usr = training_function(ml_ss, support_set_x, support_set_y, query_set_x, query_set_y,user_dynamics[user])
                    training_loss.append(loss)
                    time_specific_user[user]=time_spec_usr
                t_loss=sum(training_loss) / len(training_loss)
                if epoch%25==0:
                    print('Meta Training Loss for epoch {}= {}'.format(epoch, t_loss))
                if t_loss >= previous_loss or epoch==300:
                    print('Meta Training Loss={}'.format(previous_loss))
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
            for user in test_user:
                support_set_x = user_data[user][0]
                support_set_y = user_data[user][1]
                query_set_x = user_data[user][2]
                query_set_y = user_data[user][3]
                loss,time_spec_usr = testing_function(ml_ss, support_set_x, support_set_y, query_set_x, query_set_y,user_dynamics[user])
                time_specific_user[user] = time_spec_usr
                testing_loss.append(loss)
                # print('Meta Test Loss for user {}= {}'.format(user, loss))
            t_loss = sum(testing_loss) / len(testing_loss)
            print('Meta Testing Loss at period {} = {}'.format(period,t_loss))
        else:
            break

    print()
