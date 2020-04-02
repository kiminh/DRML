import pandas as pd
import pickle

path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
filename=['combined_data_1','combined_data_2','combined_data_3','combined_data_4']
for names in filename:
    data_file= open("{}/{}.txt".format(path,names), "r")
    print('-----Reading of a file is done-----')
    data_list=[l for l in data_file.readlines()]
    print('-----linewise reading of the file is done-----')
    f_list=[]
    mov_id=1
    for line in data_list:
        line=line.split('\n')[0]
        s_list=[]
        if line.find(':')!=-1:
            mov_id=line.split(':')[0]
            continue
        else:
            s_list.append(mov_id)
            other=line.split(',')
            if int(other[2].split('-')[0])<2002:
                continue

            for l in other:
                s_list.append(l)
            year=int(other[2].split('-')[0])
            month=int(other[2].split('-')[1])
            if year==2002:
                if month<=3:
                    s_list.append('1')
                elif month > 3 and month <= 6:
                    s_list.append('2')
                elif month > 6 and month <= 9:
                    s_list.append('3')
                else:
                    s_list.append('4')
            elif year==2003:
                if month<=3:
                    s_list.append('5')
                elif month > 3 and month <= 6:
                    s_list.append('6')
                elif month > 6 and month <= 9:
                    s_list.append('7')
                else:
                    s_list.append('8')
            elif year==2004:
                if month<=3:
                    s_list.append('9')
                elif month > 3 and month <= 6:
                    s_list.append('10')
                elif month > 6 and month <= 9:
                    s_list.append('11')
                else:
                    s_list.append('12')
            else:
                if month<=3:
                    s_list.append('13')
                elif month > 3 and month <= 6:
                    s_list.append('14')
                elif month > 6 and month <= 9:
                    s_list.append('15')
                else:
                    s_list.append('16')

        f_list.append(s_list)

    print('{}.txt file is converted and added periods as well.'.format(names))

    with open("{}/{}.csv".format(path,names), "w") as file:
        for line in f_list:
            c=0
            for word in line:
                file.write(str(word))
                if c<(len(line)-1):
                    file.write(',')
                    c=c+1
            file.write('\n')
    file.close()

    print('{}.csv file is generated.'.format(names))

#read all csv data
rating_data_path = "{}/combined_data_1.csv".format(path)
rating_data1 = pd.read_csv(
    rating_data_path, names=['movie_id', 'user_id', 'rating', 'timestamp','period'],
            sep=",", engine='python'
        )

print('-----Reading of first csv file is done-----')

rating_data_path = "{}/combined_data_2.csv".format(path)
rating_data2 = pd.read_csv(
    rating_data_path, names=['movie_id', 'user_id', 'rating', 'timestamp','period'],
            sep=",", engine='python'
        )

print('-----Reading of seocnd csv file is done-----')

rating_data_path = "{}/combined_data_3.csv".format(path)
rating_data3 = pd.read_csv(
    rating_data_path, names=['movie_id', 'user_id', 'rating', 'timestamp','period'],
            sep=",", engine='python'
        )

print('-----Reading of third csv file is done-----')

rating_data_path = "{}/combined_data_4.csv".format(path)
rating_data4 = pd.read_csv(
    rating_data_path, names=['movie_id', 'user_id', 'rating', 'timestamp','period'],
            sep=",", engine='python'
        )

print('-----Reading of fourth csv file is done-----')


user_dict = {}
label_dict = {}
rating_data=pd.concat([rating_data1,rating_data2,rating_data3,rating_data4])
print('---combined dataframe created----')

unique_user=rating_data.user_id.unique()
print('length of unique user=')
print(len(unique_user))

#user and label dictionary computation
user_df=pd.DataFrame()
df_list=[]
cnt=0
for user_idx in unique_user:
    user_temp = {}
    label_temp = {}
    temp_score = rating_data[rating_data['user_id'] == user_idx]
    cnt=cnt+1
    if cnt%10000==0:
        print(cnt)
    if len(temp_score.period.unique())<16:
        continue
    else:
        df_list.append(temp_score)
        user_df=pd.concat([user_df,temp_score])
        for period in range(1, 17):
           period_score=temp_score[temp_score['period']==period]
           p_item = period_score['movie_id'].values
           p_rating = period_score['rating'].values
           if len(p_item)<1:
               continue
           user_temp[period]=p_item
           label_temp[period]=p_rating
    user_dict[user_idx] =user_temp
    label_dict[user_idx] = label_temp
user_df=pd.concat([l for l in df_list],ignore_index=True)
user_df.to_csv("{}/rating_user_for_all_period__data.csv".format(path))
pickle.dump(user_dict, open("{}/user_dict.pkl".format(path), "wb"))
pickle.dump(label_dict, open("{}/label_dict.pkl".format(path), "wb"))
print('-----user interacted movies per period per user is pickled and dumped-----')

#active user computation
user_dict = pickle.load(open("{}/user_dict.pkl".format(path), "rb"))
label_dict = pickle.load(open("{}/label_dict.pkl".format(path), "rb"))

cnt=1
active_user=[]
for user,item in user_dict.items():
    if cnt%100==0:
        print(cnt)
    cn=1
    for period,movie in item.items():
        movie_list=[]
        if len(movie)<8 or len(movie)>100:
            cn=0
            break
    if cn==1:
        active_user.append(user)
    cnt+=1

active_user_dict={}
active_label_dict={}
for user in active_user:
    active_user_dict[user]=user_dict[user]
    active_label_dict[user]=label_dict[user]
pickle.dump(active_user_dict, open("{}/active_user_dict.pkl".format(path), "wb"))
pickle.dump(active_label_dict, open("{}/active_user_label_dict.pkl".format(path), "wb"))