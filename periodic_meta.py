import pickle
import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np

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


# plot function
def plot_function(x, y,title):
    plt.plot(x, y, 'r--')
    plt.yticks(np.arange(0, max(y) + 0.5, 0.5))
    plt.xticks(np.arange(0, max(x) + 1, 5))
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.title(title)
    plt.show()


#simple nn
class simple_neural_network(torch.nn.Module):
    def __init__(self,input_dim):
        super(simple_neural_network, self).__init__()
        self.use_cuda='True'
        self.fc1 = nn.Linear(input_dim, 64)
        self.i2o = nn.Linear(64,input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        hidden_out=self.fc1(input)
        hidden_out=F.relu(hidden_out)
        output = self.i2o(hidden_out)
        output=self.sigmoid(output)
        return output


class simple_meta_learning(torch.nn.Module):
    def __init__(self):
        super(simple_meta_learning, self).__init__()
        self.use_cuda ='True'
        self.model = simple_neural_network(96)
        self.local_lr = 1e-4
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        for idx in range(num_local_update):
            user_rep = self.model(support_set_x)
            user_rep=torch.mean(user_rep,0)
            support_set_y_pred=torch.matmul(support_set_x, user_rep.t())
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
            support_set_xs=support_set_xs.cuda()
            support_set_ys=support_set_ys.cuda()
            query_set_xs=query_set_xs.cuda()
            query_set_ys=query_set_ys.cuda()
        query_set_y_pred = self.forward(support_set_xs, support_set_ys, query_set_xs, num_local_update)
        loss_q = criterion(query_set_y_pred.view(-1, 1), query_set_ys)
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        return loss_q

def dataset_prep(mov_list,movie_dict):
    data_tensor = []
    for mov in mov_list:
        movie_info = movie_dict[mov]
        data_tensor.append(movie_info.float())
    return torch.stack(data_tensor)

def training_function(ml_ss,support_set_x,support_set_y,query_set_x, query_set_y):
    training_loss = []

    for support_tensor, support_label, query_tensor, query_label in zip(support_set_x, support_set_y,
                                                                         query_set_x, query_set_y):
        user_loss = ml_ss.global_update(support_tensor, support_label, query_tensor, query_label, 1)
        training_loss.append(user_loss)
    loss = sum(training_loss) / len(training_loss)
    return loss

def testing_function(ml_ss,test_sup_set_x, test_sup_set_y, test_que_set_x, test_que_set_y):
    testing_loss = []

    for support_tensor, support_label, query_tensor, query_label in zip(test_sup_set_x, test_sup_set_y,
                                                                        test_que_set_x, test_que_set_y):
        user_loss = ml_ss.global_update(support_tensor, support_label, query_tensor, query_label, 0)
        testing_loss.append(user_loss)
    loss = sum(testing_loss) / len(testing_loss)
    return loss

def data_generation(active_user_dict,active_label_dict,movie_dict):
    support_set_x = []
    support_set_y = []
    query_set_x = []
    query_set_y = []
    test_sup_set_x = []
    test_sup_set_y=[]
    test_que_set_x = []
    test_que_set_y=[]
    for user, item, labels in zip(active_user_dict.keys(), active_user_dict.values(), active_label_dict.values()):
        train_movie = []
        train_label = []

        for period in range(1, 16):
            train_movie.append(item[period])
            train_label.append(labels[period])
        train_movie = [l for sublist in train_movie for l in sublist]
        train_label = [l for sublist in train_label for l in sublist]

        support_tensor = dataset_prep(train_movie[15:],movie_dict)
        support_set_x.append(support_tensor)

        support_label = torch.unsqueeze(torch.tensor(train_label[15:]).float(), 1)
        support_set_y.append(support_label)
        query_label = torch.unsqueeze(torch.tensor(train_label[:15]).float(), 1)
        query_set_y.append(query_label)

        query_tensor = dataset_prep(train_movie[:15],movie_dict)
        query_set_x.append(query_tensor)

        test_movie = item[16]
        test_label = labels[16]
        test_sup_tensor = dataset_prep(test_movie[5:],movie_dict)
        test_sup_set_x.append(test_sup_tensor)
        test_que_tensor = dataset_prep(test_movie[:5],movie_dict)
        test_que_set_x.append(test_que_tensor)
        test_support_label = torch.unsqueeze(torch.tensor(test_label[5:]).float(), 1)
        test_sup_set_y.append(test_support_label)
        test_query_label = torch.unsqueeze(torch.tensor(test_label[:5]).float(), 1)
        test_que_set_y.append(test_query_label)
    return support_set_x,support_set_y,query_set_x, query_set_y,\
           test_sup_set_x,test_sup_set_y,test_que_set_x,test_que_set_y

#main fumction
if __name__ == "__main__":
    device=torch.device('cpu')
    path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
    active_user_dict = pickle.load(open("{}/final_user_dict.pkl".format(path), "rb"))
    active_label_dict = pickle.load(open("{}/final_label_dict.pkl".format(path), "rb"))
    movie_dict = pickle.load(open("{}/movie_dict.pkl".format(path), "rb"))

    support_set_x, support_set_y, query_set_x, query_set_y, test_sup_set_x, test_sup_set_y, \
    test_que_set_x, test_que_set_y=data_generation(active_user_dict,active_label_dict,movie_dict)

    ml_ss = simple_meta_learning()
    ml_ss.cuda()
    ml_ss.train()
    epoch = 1
    previous_loss = 999

    #Meta training
    training_loss=[]
    x_tick=[]
    while epoch <= 50:
        loss=training_function(ml_ss,support_set_x,support_set_y,query_set_x, query_set_y)
        print('Meta Training Loss at epoch {}= {}'.format(epoch, loss))
        if loss>=previous_loss:
            break
        else:
            previous_loss=loss
            training_loss.append(loss)
            x_tick.append(epoch)
            epoch+=1

    y=[l for l in training_loss ]
    y=torch.stack(y).to(device).detach().numpy()
    # plot_function(x_tick,y,'Meta Training Loss with the number of Local update is 1')
    #Meta Test
    loss=testing_function(ml_ss,test_sup_set_x, test_sup_set_y, test_que_set_x, test_que_set_y)
    print('Meta Testing Loss = {}'.format( loss))

