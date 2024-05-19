import json
import h5py
import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mat73

parser = argparse.ArgumentParser(
    description='Generate keyshots, keyframes and score bar.')
parser.add_argument('--h5_path', type=str,
                    help='path to hdf5 file that contains information of a dataset.', default='tvsum.h5')
parser.add_argument('-j', '--json_path', type=str,
                    help='path to json file that stores pred score output by model, it should be saved in score_dir.',
                    default='score_dir_tvsum/epoch-4.json')
parser.add_argument('-r', '--data_root', type=str,
                    help='path to directory of original dataset.', default='ydata-tvsum50-v1_1')
parser.add_argument('-s', '--save_dir', type=str,
                    help='path to directory where generating results should be saved.', default='Results_tvsum')
parser.add_argument('-d', '--dataset', type=str,
                    help='which dataset videos you want to summarize?', default='tvsum')

parser.add_argument('--summary_rate', type=float, default=0.3)

args = parser.parse_args()
h5_path = args.h5_path
json_path = args.json_path
data_root = args.data_root
save_dir = args.save_dir
dataset = args.dataset.lower()

if dataset == 'tvsum':
    video_dir = os.path.join(data_root, 'ydata-tvsum50-video', 'video')
    matlab_path = os.path.join(
        data_root, 'ydata-tvsum50-matlab', 'matlab', 'ydata-tvsum50.mat')
    d = mat73.loadmat(matlab_path)
    map_dict = {}
    for i in range(len(d['tvsum50']['video'])):
        map_dict[i + 1] = d['tvsum50']['video'][i]
    print(map_dict)

elif dataset == 'summe':
    video_dir = os.path.join(data_root, 'videos')
    matlab_path = os.path.join(data_root, 'GT')
    map_dict = {}
    i = 0
    for filename in os.listdir(matlab_path):
        map_dict[i + 1] = filename[:-4]
        i += 1
f_data = h5py.File(h5_path)
with open(json_path) as f:
    json_dict = json.load(f)
    ids = json_dict.keys()

def knapsack(v, w, W):
    r = len(v) + 1
    c = W + 1

    v = np.r_[[0], v]
    w = np.r_[[0], w]

    dp = [[0 for i in range(c)] for j in range(r)]

    for i in range(1, r):
        for j in range(1, c):
            if w[i] <= j:
                dp[i][j] = max(v[i] + dp[i-1][j-w[i]], dp[i-1][j])
            else:
                dp[i][j] = dp[i-1][j]

    chosen = []
    i = r - 1
    j = c - 1
    while i > 0 and j > 0:
        if dp[i][j] != dp[i-1][j]:
            chosen.append(i-1)
            j = j - w[i]
            i = i - 1
        else:
            i = i - 1

    return dp[r-1][c-1], chosen

def upsample(down_arr, vidlen):
    up_arr = np.zeros(vidlen)
    d_length = len(down_arr)
    ratio = vidlen // d_length
    l = (vidlen - ratio * d_length) // 2
    i = 0
    while i < d_length:
        up_arr[l:l+ratio] = np.ones(ratio, dtype=int) * down_arr[i][0]
        l += ratio
        i += 1

    return up_arr

def select_keyshots(video_info, pred_score):
    vidlen = video_info['n_frames'][()]
    cps = video_info['change_points'][:]
    weight = video_info['n_frame_per_seg'][:]
    pred_score = np.array(pred_score)
    pred_score = upsample(pred_score, vidlen)
    pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
    _, selected = knapsack(pred_value, weight, int(0.15 * vidlen))
    selected = selected[::-1]
    key_labels = np.zeros((vidlen,))
    for i in selected:
        key_labels[cps[i][0]:cps[i][1]] = 1

    return pred_score.tolist(), selected, key_labels.tolist()

selected_dict = {}

def get_keys(id):
    lst_id = id.split('_')
    video_info = f_data[id]
    video_path = os.path.join(video_dir, map_dict[int(lst_id[1])] + '.mp4')
    cps = video_info['change_points'][()]
    pred_score = json_dict[id]
    _, pred_selected, pred_summary = select_keyshots(video_info, pred_score)
    print(f'id = {id}')
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()
    frames = np.array(frames)
    keyshots = []
    indices = []
    for sel in pred_selected:
        for i in range(cps[sel][0], cps[sel][1]):
            keyshots.append(frames[i])
            indices.append(i)
    keyshots = np.array(keyshots)
    selected_dict[id] = indices
    write_path = os.path.join(save_dir, id, 'summary.avi')
    video_writer = cv2.VideoWriter(
        write_path, cv2.VideoWriter_fourcc(*'XVID'), 24, keyshots.shape[2:0:-1])
    for frame in keyshots:

        video_writer.write(frame)
    video_writer.release()

def gen_summary():
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for id in ids:
        os.mkdir(os.path.join(save_dir, id))
        get_keys(id)


if __name__ == '__main__':
    plt.switch_backend('agg')
    gen_summary()
    with open('D:\\Download\\DATN\\Video-Summarization-GAN-AED\\data\\score_anno\\select_frame_tvsum.json', 'w') as f:
        json.dump(selected_dict, f)

f_data.close()
'''
python generate_video_summary.py --h5_path D:\Download\DATN\Video-Summarization-GAN-AED\data\TVSum\eccv16_dataset_tvsum_google_pool5.h5 --j D:\Download\DATN\Video-Summarization-GAN-AED\SUM-GAN-AED\exp1\tvsum\results\split4\tvsum_16.json -r D:\Download\DATN\tvsum50_ver_1_1\ydata-tvsum50-v1_1\ -s D:\Download\DATN\test_output2\ -d tvsum
'''