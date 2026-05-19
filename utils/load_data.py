import os
import pickle
import numpy as np

# 当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 各数据集对应的接收机索引
rx_indexes_of_manysig = [
    '1-1', '1-19', '2-1', '2-19', '3-19', '7-7', '7-14',
    '8-8', '14-7', '18-2', '19-2', '20-1'
]

rx_indexes_of_manyrx = [
    '1-1', '1-19', '1-20', '2-1', '2-19', '3-19',
    '7-7', '7-14', '8-7', '8-8', '8-14',
    '13-7', '13-14', '14-7',
    '18-2', '18-19',
    '19-1', '19-2', '19-19', '19-20',
    '20-1', '20-19', '20-20',
    '23-1', '23-3', '23-5', '23-6', '23-7',
    '24-5', '24-6', '24-13', '24-16'
]

def preprocessing(x):
    """
    对输入数据进行归一化，使每个样本的功率为1
    """
    for i in range(x.shape[0]):
        power = np.sum(x[i, 0, :]**2 + x[i, 1, :]**2) / x.shape[2]
        x[i] = x[i] / np.sqrt(power)
    return x

def load_single_dataset(dataset, rx_index, date_index, tx_num, is_eq):
    """
    加载单个 Rx/日期的数据，提取每个 Tx 的前100个样本并标准化
    """
    if dataset == 'ManySig':
        rx_indexes = rx_indexes_of_manysig
    elif dataset == 'ManyRx':
        rx_indexes = rx_indexes_of_manyrx
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    try:
        folder_path = os.path.join(current_dir, '..', 'dataset', f'{dataset}/{is_eq}')
        file_path = os.path.join(folder_path, f'date{date_index}/rx_{rx_indexes[rx_index]}_data.pkl')

        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at path: {file_path}")

    x_list, y_list = [], []

    for tx_index in range(tx_num):
        tx_data = data['data'][tx_index]  # shape: (N, 2, L)
        tx_data_formatted = np.transpose(tx_data, (0, 2, 1))[:100]  # shape: (100, L, 2)
        x_list.append(tx_data_formatted)
        y_list.extend([tx_index] * tx_data_formatted.shape[0])

    x = np.concatenate(x_list, axis=0)  # shape: (N_total, L, 2)
    x = preprocessing(x)                # shape: (N_total, L, 2) normalized
    y = np.array(y_list)

    return x, y