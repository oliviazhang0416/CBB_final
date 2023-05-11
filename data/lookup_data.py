import pandas as pd
import numpy as np

data = pd.read_csv('final_without_normalization.csv')
labels = data['r-90_days_survival'].values
print(len(labels),len(data))

all_label = []
all_seq = []
all_length = []
for index,iter_data in enumerate(data.drop(['r-90_days_survival'],axis=1).iterrows()):
    item = dict(iter_data[1])
    all_seq.append(list(item.values()))
    if len(item.values()) not in all_length:
        all_length.append(len(item.values()) )
    all_label.append(labels[index])

all_label = np.array(all_label)
all_seq = np.array(all_seq)

per1 = np.random.permutation(all_seq.shape[0])
print(per1)
samp_arr = all_seq[per1]
samp_label = all_label[per1]

train_seq = samp_arr[:int(0.8*len(samp_arr))]
test_seq = samp_arr[int(0.8*len(samp_arr)):]


train_label = samp_label[:int(0.8*len(samp_label))]
test_label = samp_label[int(0.8*len(samp_label)):]
print(type(train_seq))
np.save('../pretrain_data/train.npy',train_seq)
np.save('../pretrain_data/train_label.npy',train_label)
np.save('../pretrain_data/test.npy',test_seq)
np.save('../pretrain_data/test_label.npy',test_label)
