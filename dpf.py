import pickle
import torch
import numpy as np

path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
active_user_dict = pickle.load(open("{}/active_user_dict.pkl".format(path), "rb"))
active_label_dict = pickle.load(open("{}/active_user_label_dict.pkl".format(path), "rb"))
movie_dict = pickle.load(open("{}/movie_dict_33.pkl".format(path), "rb"))


#model
def dpf_model(u_nk,u_nkt,q_i):
    user_expo=torch.exp(torch.add(u_nk,u_nkt))
    item_expo=torch.exp(q_i)
    y_nmt= user_expo*item_expo
    y_nmt=y_nmt[:,-1,:]
    return torch.mean(y_nmt,dim=1)


#loss function
def loss_function_mse(r_uit,r_pre):
    return (r_uit-r_pre)**2

#grad of b_u and b_i
def grad_unkt(y_pred,y_true):
    return 2*torch.mean(y_true-y_pred)

mu,sigma=0,0.1
number_latent_factor=33
u_nk=np.random.normal(mu, sigma, number_latent_factor)
#v_mk=np.random.normal(mu, sigma, number_latent_factor)
for period in range(1,17):
    if period==1:
        u_nkt = np.random.normal(mu, sigma, number_latent_factor)
        #v_mkt = np.random.normal(mu, sigma, number_latent_factor)
    else:
        u_nkt = np.random.normal(np.mean(u_nkt), sigma, number_latent_factor)
        #v_mkt = np.random.normal(np.mean(v_mkt), sigma, number_latent_factor)

    periodic_loss = []
    u_nk = torch.from_numpy(u_nk).float()
    u_nkt = torch.from_numpy(u_nkt).float()
    for user, movie in active_user_dict.items():
        train_tensor = []
        mov = []
        train_label = []
        for i in range(period):
            mov.append(movie[i + 1])
            train_label.append(active_label_dict[user][i + 1])
        mov = [l for sublist in mov for l in sublist]
        train_label = [l for sublist in train_label for l in sublist]
        for mo in mov:
            movie_info = movie_dict[mo]
            train_tensor.append(movie_info.float())

        train_tensor = torch.stack(train_tensor)
        train_label = torch.unsqueeze(torch.tensor(train_label).float(), 1)
        prev_loss = 9999

        while True:
            y_pred = dpf_model(u_nk, u_nkt,train_tensor)
            y_pred = y_pred.view(-1, 1)
            loss = loss_function_mse(train_label, y_pred)
            mean_loss = torch.mean(loss)
            print(mean_loss)
            u_ug = grad_unkt(y_pred, train_label)
            rate = 0.001
            u_nkt = u_nkt - rate * u_ug
            if prev_loss <= mean_loss:
                periodic_loss.append(prev_loss)
                break
            prev_loss = mean_loss

    losses = torch.mean(torch.stack(periodic_loss))

    print('Period {} Loss={}'.format(period, losses))

print()