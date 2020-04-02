import pickle
import torch
import numpy as np

path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
active_user_dict = pickle.load(open("{}/active_user_dict.pkl".format(path), "rb"))
active_label_dict = pickle.load(open("{}/active_user_label_dict.pkl".format(path), "rb"))
movie_dict = pickle.load(open("{}/movie_dict_33.pkl".format(path), "rb"))


#user preference
def user_preference_x_it(mu_n, trans_mat, cov_mat, alpha, sigma_n,y_label):
    b_n=alpha+cov_mat
    b_n=torch.inverse(b_n)
    sigma_nplus=torch.inverse(torch.matmul(trans_mat,trans_mat.t())/sigma_n +b_n)
    temp=[]
    y_label=torch.tensor(y_label,dtype=float).view(-1,1)
    for label in y_label:
        label=torch.reshape(label,(1,1))
        temp.append(trans_mat.t()*label / sigma_n)
    stack_temp=torch.stack(temp)
    mean_temp=torch.mean(stack_temp,dim=0)
    final_mean=torch.mean(mean_temp,0)
    bn_mul=torch.matmul(b_n,mu_n.t())
    sum_bnm=final_mean+bn_mul
    mu_nplus=torch.matmul(sum_bnm,sigma_nplus.t())
    return mu_nplus

#CKF model
def ckf_model(factor_dim):
    user_factor={}
    user_k={}
    trans_mat=torch.eye(factor_dim,factor_dim)
    cov_mat=torch.eye(factor_dim,factor_dim)
    u_init=torch.zeros(factor_dim)
    alpha=0.01
    sigma=0.001
    for period in range(1,17):
        for user in active_user_dict.keys():
            user_k[user]=u_init
        user_factor[period]=user_k


    for user,item in active_user_dict.items():
        user_init_factor = u_init
        for period in range(1,17):
            if period<16:
                ite=0
                while True:
                    if ite==5:
                        break
                    user_init_factor = user_preference_x_it(user_init_factor, trans_mat, cov_mat, alpha,
                                                            sigma,active_label_dict[user][period])
                    ite+=1
                user_factor[period][user] = user_init_factor
            else:
                test_tensor = []
                for mov in item[period]:
                    movie_info = movie_dict[mov]
                    test_tensor.append(movie_info.float())
                test_tensor = torch.stack(test_tensor)
                y_label=active_label_dict[user][period]
                #test_label = torch.unsqueeze(torch.tensor(test_label).float(), 1)
                y_pred=torch.matmul(user_factor[period-1][user].t(),test_tensor)
                print('---True Labels---')
                print(y_label)
                print('Predicted')
                print(y_pred)


ckf_model(33)
