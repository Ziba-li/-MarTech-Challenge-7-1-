import os
import pickle
import numpy as np
from dateutil import parser
from loguru import logger
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm
import pandas as pd

np.random.seed(10)
SUBMISSION_PATH = '../data/submission.csv'
TRAIN_PATH = '../data/train.csv'
pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', 200)
start_date = '2010-01-01 00:00:00'


def last_diff(n):
    def func(row):
        data = list(row)
        if len(data) < n + 1:
            return 365 * 24 * 3600
        e = parser.parse(data[-1])
        s = parser.parse(data[-n - 1])
        ret = (e - s).total_seconds()
        return ret

    return func


def count_filed_features(data, filed):
    data[[f'{filed}_min',
          f'{filed}_max',
          f'{filed}_mean',
          f'{filed}_sum',
          f'{filed}_std']] = data.groupby('customer_id')[filed].agg([(f'{filed}_min', 'min'),
                                                                     (f'{filed}_max', 'max'),
                                                                     (f'{filed}_mean', 'mean'),
                                                                     (f'{filed}_sum', 'sum'),
                                                                     (f'{filed}_std', 'std')])
    data[f'{filed}_std'] = data[f'{filed}_std'].fillna(0)
    return data


def frequent_month(n, max_month):
    def func(row):
        data = list(row)
        c = 0
        for i in data:
            m = int(i.split('-')[1])
            if m == max_month - n:
                c += 1
        return c

    return func


def tail_n(n):
    def func(row):
        data = list(row)
        if len(data) < n:
            return -1
        return data[-n]

    return func


def get_order_tail(data, ret, filed):
    tmp_data = data.groupby('order_id')['customer_id', filed].last()
    for i in range(2, 10):
        ret[f'{filed}_tail_{i}'] = tmp_data.groupby('customer_id')[filed].agg(tail_n(i))
    return ret


def get_tail(data, ret, filed):
    for i in range(2, 10):
        ret[f'{filed}_tail_{i}'] = data.groupby('customer_id')[filed].agg(tail_n(i))
    return ret


def field_top1(row):
    data = list(row)
    d = {}
    for i in data:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    m = 0
    ret = ''
    for k, v in d.items():
        if v > m:
            ret = k
            m = k
    return ret


def time_convert(data, field_name, max_date):
    ret = data
    ret[field_name] = pd.to_datetime(ret[field_name])
    ret[f'{field_name}_year'] = ret[field_name].dt.year
    ret[f'{field_name}_month'] = ret[field_name].dt.month
    ret[f'{field_name}_day'] = ret[field_name].dt.day
    ret[f'{field_name}_hour'] = ret[field_name].dt.hour
    min_time = ret[field_name].min()
    ret[f"{field_name}_day_diff"] = (ret[field_name] - min_time).dt.days
    ret[f'{field_name}_rfm_r'] = (max_date - ret[field_name]).dt.days
    ret[field_name] = ret[field_name].values.astype(np.int64) / 1000000000
    print(field_name, "min time", min_time)
    return ret


def label_map(data_map):
    l_map = {}
    i = 0
    for k, v in data_map.items():
        l_map[k] = i
        i += 1
    return l_map


def get_data():
    logger.info("get data")
    raw = pd.read_csv(TRAIN_PATH)
    submit_data = pd.read_csv(SUBMISSION_PATH)
    raw.sort_values('order_pay_time', ascending=True, inplace=True)
    logger.info("load finish")
    label_raw = set(raw[raw['order_pay_time'] > '2013-07-31 23:59:59']['customer_id'])
    return raw, submit_data, label_raw


def get_feature(train_raw, day_time):
    max_month = int('2013-07-31 23:59:59'.split('-')[1]) + 1
    max_date = parser.parse('2013-07-31 23:59:59')
    train_raw = train_raw[(train_raw['order_pay_time'] <= day_time) & (train_raw['order_pay_time'] >= start_date)]
    ret = pd.DataFrame(train_raw.groupby('customer_id')['customer_gender'].last().fillna(0))  # 包含去除重复的customer_id
    train_raw["goods_price"] = train_raw["goods_price"].fillna(0)

    ret[['goods_id', 'goods_status', 'goods_price', 'goods_has_discount', 'goods_list_time', 'goods_delist_time',
         'member_id', 'member_status', 'is_member_actived', 'goods_class_id', 'order_total_num', 'order_amount',
         'order_total_payment', 'order_total_discount', 'order_pay_time', 'order_status', 'order_count',
         'is_customer_rate', 'order_detail_status', 'order_detail_goods_num', 'order_detail_amount',
         'order_detail_payment', 'order_detail_discount']] = train_raw.groupby('customer_id')[
        'goods_id', 'goods_status', 'goods_price', 'goods_has_discount', 'goods_list_time', 'goods_delist_time',
        'member_id', 'member_status', 'is_member_actived', 'goods_class_id', 'order_total_num', 'order_amount',
        'order_total_payment', 'order_total_discount', 'order_pay_time', 'order_status', 'order_count',
        'is_customer_rate', 'order_detail_status', 'order_detail_goods_num', 'order_detail_amount',
        'order_detail_payment', 'order_detail_discount'].last()

    ret = count_filed_features(ret, "goods_price")
    ret = count_filed_features(ret, "order_total_payment")
    ret = count_filed_features(ret, "order_detail_discount")

    ret[['goods_cnt']] = train_raw.groupby('customer_id')['goods_id'].agg([('goods_cnt', 'count')])
    ret[['order_cnt']] = train_raw.groupby('customer_id')['order_id'].agg([('goods_cnt', 'count')])
    ret[['customer_province', 'customer_city']] = train_raw.groupby('customer_id')[
        'customer_province', 'customer_city'].last()
    ret[['is_customer_rate_mean', 'is_customer_rate_sum']] = train_raw.groupby('customer_id')['is_customer_rate'].agg(
        [('is_customer_rate_mean', 'mean'), ('is_customer_rate_sum', 'sum')])
    ret[['order_total_num_mean', 'order_total_num_sum']] = train_raw.groupby('customer_id')['order_total_num'].agg(
        [('order_total_num_mean', 'mean'), ('order_total_num_sum', 'sum')])
    ret[['goods_has_discount_mean', 'goods_has_discount_sum']] = train_raw.groupby('customer_id')[
        'goods_has_discount'].agg(
        [('goods_has_discount_mean', 'mean'), ('goods_has_discount_sum', 'sum')])

    tmp_data = train_raw.groupby('order_id')['customer_id', 'order_pay_time'].last()
    for i in range(1, 6):
        ret[f'order_pay_time_last_diff_{i}'] = tmp_data.groupby('customer_id')['order_pay_time'].agg(last_diff(i))

    tmp_data = train_raw.groupby('order_id')['customer_id', 'order_pay_time'].last()
    for i in range(1, 6, 1):
        ret['order_pay_time_frequent_tail_{}'.format(i)] = tmp_data.groupby('customer_id')['order_pay_time'].agg(
            frequent_month(i, max_month))
    return train_raw, ret, max_date


def add_feature(data, ret, max_date):
    ret = get_order_tail(data, ret, 'order_total_payment')
    ret = get_order_tail(data, ret, 'order_total_discount')
    ret = get_order_tail(data, ret, 'order_amount')
    ret = get_order_tail(data, ret, 'order_total_num')
    ret = get_order_tail(data, ret, 'order_count')

    ret = get_tail(data, ret, 'goods_id')
    ret = get_tail(data, ret, 'goods_class_id')
    ret = get_tail(data, ret, 'is_customer_rate')
    ret = get_tail(data, ret, 'goods_has_discount')

    ret['goods_class_id_top1'] = data.groupby('customer_id')['goods_class_id'].agg(field_top1)
    ret['goods_id_top1'] = data.groupby('customer_id')['goods_id'].agg(field_top1)
    ret = ret.reset_index()
    ret = time_convert(ret, 'order_pay_time', max_date)
    ret = time_convert(ret, 'goods_list_time', max_date)
    ret = time_convert(ret, 'goods_delist_time', max_date)
    ret['good_show_days'] = (ret['goods_delist_time'] - ret['goods_list_time']) / (24 * 60 * 60)
    ret["customer_province"] = ret["customer_province"].fillna(0)
    ret["customer_city"] = ret["customer_city"].fillna(0)
    ret["member_status"] = ret["member_status"].fillna(0)
    ret["is_member_actived"] = ret["is_member_actived"].fillna(0)

    p_map = label_map(dict(ret['customer_province'].value_counts()))
    c_map = label_map(dict(ret['customer_city'].value_counts()))
    ret['customer_province'] = ret['customer_province'].map(p_map)
    ret['customer_city'] = ret['customer_city'].map(c_map)

    good_id_map, good_class_map = data['goods_id'].value_counts(), data['goods_class_id'].value_counts()
    ret['good_id_total'] = ret['goods_id'].map(dict(good_id_map))
    ret['goods_class_total'] = ret['goods_class_id'].map(dict(good_class_map))

    drop_features = ['goods_price_mean', 'goods_price_std', 'goods_price_sum', 'order_detail_discount_std',
                     'order_total_payment_sum', 'order_total_payment_std', 'order_detail_discount_mean',
                     'order_detail_discount_sum', 'order_total_payment_mean']

    for i in drop_features:
        ret = ret.drop(i, axis=1)
    return ret


def pickle_file(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file_path):
    if not os.path.exists(file_path):
        return
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def process():
    raw, submit_data, label_raw = get_data()

    train_raw1 = raw[(raw['order_pay_time'] <= '2013-07-31 23:59:59') & (raw['order_pay_time'] >= start_date)]
    train_raw2, ret2, max_date2 = get_feature(train_raw1, '2013-07-31 23:59:59')
    train_df = add_feature(train_raw2, ret2, max_date2)
    pickle_file(train_df, "train_df.pk")

    train_raw, ret, max_date = get_feature(raw[raw['order_pay_time'] >= start_date], '2013-08-31 23:59:59')
    all_data_fit_df = add_feature(train_raw, ret, max_date)
    pickle_file(all_data_fit_df, "all_data_fit_df.pk")

    submit_data.value_counts('result')
    label = train_df['customer_id'].map(lambda x: 1 if x in label_raw else 0)
    label = label.to_numpy()

    norm_model = preprocessing.StandardScaler().fit(train_df.to_numpy())  # 标准差
    train = norm_model.transform(train_df.to_numpy())
    all_data_fit = norm_model.transform(all_data_fit_df.to_numpy())

    pickle_file(train, "train.pk")
    pickle_file(all_data_fit, "all_data_fit.pk")
    pickle_file(all_data_fit_df, "all_data_fit_df.pk")
    pickle_file(label, "label.pk")


def get_best_threshold(y_actual, y_pred):
    m_auc = 0
    t_opt = -1.0
    up, down = 0.03, 0.015
    step = 0.001
    t = up
    while t > down:
        ret = [1 if x > t else 0 for x in y_pred]
        auc = roc_auc_score(y_actual, ret)
        if auc > m_auc:
            m_auc = auc
            t_opt = t
            print(m_auc, t_opt)
        t -= step
    return t_opt, m_auc


def model_desc(model, x, y):
    y_pred = model.predict(x)
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    print("proba auc", auc)
    optimal_th, optimal_point = find_optimal_cutoff(TPR=fpr, FPR=tpr, threshold=thresholds)
    print("yoden", optimal_point, optimal_th)

    def get_best_th():
        optimal_th, auc = get_best_threshold(y, y_pred)
        ret = [1 if x > optimal_th else 0 for x in y_pred]
        auc = roc_auc_score(y, ret)
        print("best auc", auc)
        print("accuracy", accuracy_score(y, ret))
        print("best sum", sum(ret), "len", len(ret))
        return optimal_th

    th = get_best_th()
    return th


def find_optimal_cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def gbm(x_train, y_train, x_test, y_test, all_data_fit, all_data_fit_df):
    tree_params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'num_leaves': 127,   # 叶子节点数
        'num_trees': 3200,  # 树个数
        'metric': 'auc',  # 评估函数
        'learning_rate': 0.001,  # 学习率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 100,
        # 'lambda_l1': 1
    }
    lgb_train = lightgbm.Dataset(x_train, y_train)
    lgb_model = lightgbm.train(tree_params, lgb_train, num_boost_round=10000, valid_sets=lgb_train)
    feature_imp = pd.DataFrame({
        'column': list(all_data_fit_df),
        'importance': lgb_model.feature_importance(),
    }).sort_values(by='importance', ascending=False)
    print(feature_imp)
    print(feature_imp.shape)

    logger.info("train")
    optimal_th = model_desc(lgb_model, x_train, y_train)
    print(f"train acc：{optimal_th}")

    logger.info("test")
    optimal_th_2 = model_desc(lgb_model, x_test, y_test)
    print(f"test acc：{optimal_th_2}")

    y_ret = lgb_model.predict(all_data_fit)
    ret = [1 if x > optimal_th else 0 for x in y_ret.flat]
    print("the final result", sum(ret), len(ret))

    test_data = pd.read_csv(SUBMISSION_PATH)
    ids = list(all_data_fit_df['customer_id'])
    ret_dict = {ids[i]: ret[i] for i in range(len(ids))}
    test_data['result'] = test_data['customer_id'].map(lambda x: ret_dict.get(x, 0))
    test_data.to_csv('gbm_1000.csv', index=False)


def run():
    process()
    logger.info("load pickle..")
    train = pickle_load("train.pk")
    all_data_fit = pickle_load("all_data_fit.pk")
    all_data_fit_df = pickle_load("all_data_fit_df.pk")
    label = pickle_load("label.pk")
    x_train, x_test, y_train, y_test = train_test_split(train, label, test_size=0.2, random_state=2021)
    gbm(x_train, y_train, x_test, y_test, all_data_fit, all_data_fit_df)


if __name__ == '__main__':
    run()
