import os
import numpy as np
import pickle

## for features extracted with 1-second sliding window
feature_1s_dir = './eeg_used_1s/'
file_1s_list = os.listdir(feature_1s_dir)
file_1s_list.sort()

train_ds = []
test_ds = []
train_labels = []
test_labels = []
for item in file_1s_list:
    npz_data = np.load( os.path.join(feature_1s_dir, item))

    # Train data
    data = pickle.loads(npz_data['train_data'])
    train_ds.append(np.swapaxes(np.array(list(data.values())), 0, 2))

    # Test data
    data = pickle.loads(npz_data['test_data'])
    test_ds.append(np.swapaxes(np.array(list(data.values())), 0, 2))

    # Train labels
    data = npz_data['train_label']
    train_labels.append(data)

    # Test labels
    data = npz_data['test_label']
    test_labels.append(data)

final_train_ds = []
final_test_ds = []
final_train_l = []
final_test_l = []


for sample, label in zip(train_ds, train_labels):
    spoof = 0
    for j in range(int(sample.shape[1] // 250)):
        final_train_ds.append(sample[:, spoof:spoof + 250, :])
        final_train_l.append(label[spoof:spoof + 250])
        spoof = spoof + 250


# open a file, where you ant to store the data
file = open('filepath', 'wb')

# dump information to that file
pickle.dump(final_train_ds, file)
pickle.dump(final_train_l, file)
pickle.dump(final_test_ds, file)
pickle.dump(final_test_l, file)

# close the file
file.close()


for item in file_1s_list:
    print('*'*50)
    print(item)
    npz_data = np.load(os.path.join(feature_1s_dir, item))
    print(list(npz_data.keys()))  # train_data, test_data, train_label, test_label
    # train_data : samples from the first 9 movie clips
    data = pickle.loads(npz_data['train_data'])
    print(data.keys())
    for kk in list(data.keys()):
        print(data[kk].shape)
    # test_data : samples from the rest 6 movie clips
    data = pickle.loads(npz_data['test_data'])
    print(data.keys())
    for kk in list(data.keys()):
        print(data[kk].shape)
    # train_label
    data = npz_data['train_label']
    print(data.shape)
    # test_label
    data = npz_data['test_label']
    print(data.shape)
    print('*'*50 + '\n')



## for features extracted with 4-second sliding window
feature_4s_dir = './eeg_used_4s/'
file_4s_list = os.listdir(feature_4s_dir)
file_4s_list.sort()

for item in file_4s_list:
    print('*'*50)
    print(item)
    npz_data = np.load( os.path.join(feature_4s_dir, item) )
    print(list(npz_data.keys()))  # train_data, test_data, train_label, test_label
    # train_data : samples from the first 9 movie clips
    data = pickle.loads(npz_data['train_data'])
    print(data.keys())
    for kk in list(data.keys()):
        print(data[kk].shape)
    # test_data : samples from the rest 6 movie clips
    data = pickle.loads(npz_data['test_data'])
    print(data.keys())
    for kk in list(data.keys()):
        print(data[kk].shape)
    # train_label
    data = npz_data['train_label']
    print(data.shape)
    # test_label
    data = npz_data['test_label']
    print(data.shape)
    print('*'*50 + '\n')



train_ds = []
test_ds = []
train_labels = []
test_labels = []
for item in file_4s_list:
    npz_data = np.load( os.path.join(feature_4s_dir, item))

    # Train data
    data = pickle.loads(npz_data['train_data'])
    train_ds.append(np.swapaxes(np.array(list(data.values())), 0, 2))

    # Test data
    data = pickle.loads(npz_data['test_data'])
    test_ds.append(np.swapaxes(np.array(list(data.values())), 0, 2))

    # Train labels
    data = npz_data['train_label']
    train_labels.append(data)

    # Test labels
    data = npz_data['test_label']
    test_labels.append(data)
