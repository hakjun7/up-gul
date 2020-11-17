import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
    
# 1
# img_paths = []
# captions = []
# with open('..\\datasets\\captions.csv', 'r') as f:
#     cs = csv.reader(f, delimiter='|')
#     for c in cs:
#         img_paths.append('.\\images\\' + c[0])
#         captions.append(c[2])
# img_paths.pop(0)
# captions.pop(0)

# # 2
# arr_img_paths = np.array(img_paths)
# arr_captions = np.array(captions)
# x_train, x_test, y_train, y_test = train_test_split(arr_img_paths, arr_captions, test_size=0.2, shuffle=True, random_state=1004)

# np.save("..\\datasets\\img_paths_train", x_train)
# np.save("..\\datasets\\captions_train", y_train)
# np.save("..\\datasets\\img_paths_test", x_test)
# np.save("..\\datasets\\captions_test", y_test)

# # 3
# trained_data = dict()
# trained_img_paths = np.load("..\\datasets\\img_paths_train.npy")
# trained_captions = np.load("..\\datasets\\captions_train.npy")
# ln = len(trained_img_paths)
# for i in range(ln):
#     if not trained_img_paths[i] in trained_data.keys():
#         trained_data[trained_img_paths[i]] = []
#     trained_data[trained_img_paths[i]].append(trained_captions[i])



# Req. 3-1	이미지 경로 및 캡션 불러오기
def get_path_caption():
    img_paths = []
    captions = []
    datasets = os.path.abspath("datasets")
    csv_path = os.path.join(datasets, 'captions.csv')
    with open(csv_path, 'r') as f:
        cs = csv.reader(f, delimiter='|')
        for c in cs:
            img_paths.append('.\\images\\' + c[0])
            captions.append(c[2])
    img_paths.pop(0)
    captions.pop(0)

    
    return img_paths, captions


# Req. 3-2	전체 데이터셋을 분리해 저장하기
def dataset_split_save(img_paths, captions):
    arr_img_paths = np.array(img_paths)
    arr_captions = np.array(captions)
    x_train, x_test, y_train, y_test = train_test_split(arr_img_paths, arr_captions, test_size=0.2, shuffle=True, random_state=1004)

    datasets = os.path.abspath("datasets")
    np.save(datasets + '\\img_paths_train', x_train)
    np.save(datasets + '\\captions_train', y_train)
    np.save(datasets + '\\img_paths_test', x_test)
    np.save(datasets + '\\captions_test', y_test)


# Req. 3-3	저장된 데이터셋 불러오기
def get_data_file(trained):
    
    datasets = os.path.abspath("datasets")
    img_paths = np.load(datasets + '\\img_paths_train.npy')
    captions = np.load(datasets + '\\captions_train.npy')
    
    if trained == False:
        img_paths = np.load(datasets + '\\img_paths_test.npy')
        captions = np.load(datasets + '\\captions_test.npy')
    
    ln = len(img_paths)
    total_data = dict()
    for i in range(ln):
        if not img_paths[i] in total_data.keys():
            total_data[img_paths[i]] = []
        total_data[img_paths[i]].append(captions[i])
    
    return total_data


# Req. 3-4	데이터 샘플링
def sampling_data():
    pass

