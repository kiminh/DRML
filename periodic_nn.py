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

def dataset_prep(mov_list,movie_dict):
    data_tensor = []
    for mov in mov_list:
        movie_info = movie_dict[mov]
        data_tensor.append(movie_info.float())
    return torch.stack(data_tensor)

def training_function(ml_ss,support_set_x,support_set_y):
    training_loss = []

    for support_tensor, support_label in zip(support_set_x, support_set_y):
        support_tensor=support_tensor.cuda()
        support_label=support_label.cuda()
        user_rep = ml_ss(support_tensor)
        user_rep = torch.mean(user_rep, 0)
        y_pred = torch.matmul(support_tensor, user_rep.t())
        loss = criterion(y_pred.view(-1, 1), support_label)
        meta_optim.zero_grad()
        loss.backward()
        meta_optim.step()
        training_loss.append(loss)
    loss = sum(training_loss) / len(training_loss)
    return loss

def testing_function(ml_ss,test_sup_set_x, test_sup_set_y):
    testing_loss = []
    for support_tensor, support_label in zip(test_sup_set_x, test_sup_set_y):
        support_tensor = support_tensor.cuda()
        support_label = support_label.cuda()
        user_rep = ml_ss(support_tensor)
        user_rep = torch.mean(user_rep, 0)
        y_pred = torch.matmul(support_tensor, user_rep.t())
        loss = criterion(y_pred.view(-1, 1), support_label)
        testing_loss.append(loss)
    loss = sum(testing_loss) / len(testing_loss)
    return loss

def data_generation(active_user_dict,active_label_dict,movie_dict):
    support_set_x = []
    support_set_y = []
    test_sup_set_x = []
    test_sup_set_y=[]

    for user, item, labels in zip(active_user_dict.keys(), active_user_dict.values(), active_label_dict.values()):
        train_movie = []
        train_label = []

        for period in range(1, 16):
            train_movie.append(item[period])
            train_label.append(labels[period])
        train_movie = [l for sublist in train_movie for l in sublist]
        train_label = [l for sublist in train_label for l in sublist]

        support_tensor = dataset_prep(train_movie,movie_dict)
        support_set_x.append(support_tensor)

        support_label = torch.unsqueeze(torch.tensor(train_label).float(), 1)
        support_set_y.append(support_label)

        train_movie = item[16]
        train_label = labels[16]

        test_sup_tensor = dataset_prep(train_movie, movie_dict)
        test_sup_set_x.append(test_sup_tensor)

        test_sup_label = torch.unsqueeze(torch.tensor(train_label).float(), 1)
        test_sup_set_y.append(test_sup_label)

    return support_set_x,support_set_y,test_sup_set_x,test_sup_set_y

#main fumction
if __name__ == "__main__":
    device=torch.device('cpu')
    path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
    active_user_dict = pickle.load(open("{}/final_user_dict.pkl".format(path), "rb"))
    active_label_dict = pickle.load(open("{}/final_label_dict.pkl".format(path), "rb"))
    movie_dict = pickle.load(open("{}/movie_dict.pkl".format(path), "rb"))

    support_set_x, support_set_y, query_set_x, \
    query_set_y=data_generation(active_user_dict,active_label_dict,movie_dict)

    model = simple_neural_network(96)
    meta_optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.cuda()
    model.train()
    epoch = 1
    previous_loss = 999

    #Meta training
    training_loss=[]
    x_tick=[]
    while epoch <= 50:

        loss=training_function(model,support_set_x,support_set_y)
        print('Training Loss at epoch {}= {}'.format(epoch, loss))
        if loss>=previous_loss:
            break
        else:
            previous_loss=loss
            training_loss.append(loss)
            x_tick.append(epoch)
            epoch+=1

    y=[l for l in training_loss ]
    y=torch.stack(y).to(device).detach().numpy()
    plot_function(x_tick,y,'Training Loss with simple Neural Network')
    #Meta Test
    loss=testing_function(model,query_set_x,query_set_y)
    print('Testing Loss = {}'.format( loss))
