import torch
import time
import datetime
from tools.ts2vec import TS2Vec
from torch.utils.data import Dataset
import numpy as np


class SalObjDataset(Dataset):
    def __init__(self, data, label):

        self.data = self.get_data_label(data, label)


    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get_data_label(self, seqs, labels):
        data = []
        for i in range(len(seqs)):
            seq = seqs[i, :]
            label = labels[i]
            data.append((seq, label))
        return data

    def get_data_seq_label(self, datas, seqs, labels):
        fea = []
        for i in range(len(seqs)):
            repre = datas[i]
            seq = torch.tensor(seqs[i], dtype=torch.float)
            label = labels[i]
            fea.append((repre, seq, label))
        return fea



def pretraining(model_path, train_data, train_label, test_data, test_label, ReTrain,  repr_dim, lr, batch_size, iters, encoding_window):

    savepath = model_path

    train_data1 = train_data[0]
    train_data2 = train_data[1]
    train_data3 = train_data[2]


    for_train = train_data1

    print('Loading data for TS2Vec ... ', end='')
    t = time.time()
    model = TS2Vec(
        input_dims=for_train.shape[-1],
        output_dims=repr_dim,
        lr=lr,
        batch_size=batch_size,
        max_train_length=3000
    )
    print('complete loading !')

    if ReTrain:
        ##########  TS2Vec   is   training  ############
        print('TS2Vec is fitting.....')
        loss_log = model.fit(
            for_train,
            n_iters=iters,
            verbose=True
        )
        t = time.time() - t
        print(f"Complete fitting! Training time: {datetime.timedelta(seconds=t)}\n")
        model.save(savepath)
    else:
        model.load(savepath)
    t = time.time()
    ###########   TS2Vec  is encodding    #############
    print('TS2Vec is encodding.....')
    #  multiscale
    train_repr1 = model.encode(train_data1, encoding_window=encoding_window if train_label[0].ndim == 1 else None)  ## fuse the T and F, but we can change it
    train_repr2 = model.encode(train_data2, encoding_window=encoding_window if train_label[1].ndim == 1 else None)  ## fuse the T and F, but we can change it
    train_repr3 = model.encode(train_data3, encoding_window=encoding_window if train_label[2].ndim == 1 else None)  ## fuse the T and F, but we can change it

    test_repr = model.encode(test_data, encoding_window=encoding_window if test_label.ndim == 1 else None)

    ts2vec_infer_time = time.time() - t
    print('Complete encodding! ts2vec_infer_time:  ', ts2vec_infer_time)
    print('see  the representation1 dimension: ', train_repr1.shape)
    print('see  the representation2 dimension: ', train_repr2.shape)
    print('see  the representation3 dimension: ', train_repr3.shape)
    train_dataset1 = SalObjDataset(train_repr1, train_label[0])
    train_dataset2 = SalObjDataset(train_repr2, train_label[1])
    train_dataset3 = SalObjDataset(train_repr3, train_label[2])

    train_dataset = [train_dataset1, train_dataset2, train_dataset3]

    val_dataset = SalObjDataset(test_repr, test_label)

    print("Pretraining is Finished.......")

    return train_dataset, val_dataset






