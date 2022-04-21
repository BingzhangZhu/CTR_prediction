# -*- coding: utf-8 -*-
from numpy import float64
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
from torch.optim import Adagrad

if __name__ == "__main__":
    # data = pd.read_csv('./raw_ad_user.csv', dtype=float)

    data = pd.read_csv('./avazu_train.csv', low_memory=False)

    # data = pd.read_csv('./criteo_sample.txt')

    # sparse_features = ['C' + str(i) for i in range(1, 26)]
    # sparse_features.append('CC')
    # dense_features = ['I' + str(i) for i in range(1, 14)]


    # sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid',
    #     'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level',  
    #     'shopping_level', 'occupation', 'new_user_class_level ', 'cate_id',
    #     'campaign_id', 'customer', 'brand']
    # dense_features = ['time_stamp', 'price']

    sparse_features = ['y' + str(i) for i in range(4, 24)]
    sparse_features.append('y1')
    dense_features = ['y3']


    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    # target = ['clk']
    target = ['y2']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                            for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DCNMix(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    task='binary', l2_reg_dnn=1e-5, dnn_hidden_units=(3, 128),
                    l2_reg_embedding=1e-5, device=device)

    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                 task='binary',
    #                 l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                    metrics=["binary_crossentropy", "auc", "acc"], )

    history = model.fit(train_model_input, train[target].values, batch_size=512, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))