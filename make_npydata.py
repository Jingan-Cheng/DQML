import os
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Dataset")
parser.add_argument("--root_dir", type=str, default="../datasets/FIDTM")
args = parser.parse_args()

if not os.path.exists('.npydata/npy'):
    os.makedirs('.npydata/npy')

'''please set your dataset path'''
shanghai_root = os.path.join(args.root_dir, "ShanghaiTech")
carpk_root = os.path.join(args.root_dir, "CARPK")
pucpr_root = os.path.join(args.root_dir, "PUCPR")
large_root = os.path.join(args.root_dir, "large-vehicle")
small_root = os.path.join(args.root_dir, "small-vehicle")
trancos_root = os.path.join(args.root_dir, "TRANCOS_v3")
ship_root = os.path.join(args.root_dir, "ship")
building_root = os.path.join(args.root_dir, "building")

try:
    shanghaiAtrain_path = shanghai_root + '/part_A_final/train_data/images/'
    shanghaiAtest_path = shanghai_root + '/part_A_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test.npy', test_list)

    print("generate ShanghaiA image list successfully")
except:
    print("The ShanghaiA dataset path is wrong. Please check you path.")

try:
    trancos_train_path = trancos_root + '/train_data/images/'
    trancos_val_path = trancos_root + '/val_data/images/'
    trancos_test_path = trancos_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(trancos_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(trancos_train_path + filename)
    train_list.sort()
    np.save('./npydata/trancos_train.npy', train_list)

    val_list = []
    for filename in os.listdir(trancos_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(trancos_val_path + filename)
    val_list.sort()
    np.save('./npydata/trancos_val.npy', val_list)

    test_list = []
    for filename in os.listdir(trancos_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(trancos_test_path + filename)
    test_list.sort()
    np.save('./npydata/trancos_test.npy', test_list)

    print("Generate trancos image list successfully")
except:
    print("The trancos dataset path is wrong. Please check your path.")

try:
    carpk_train_path = carpk_root + '/train_data/images/'
    carpk_test_path = carpk_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(carpk_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(carpk_train_path + filename)
    train_list.sort()
    np.save('./npydata/carpk_train.npy', train_list)

    test_list = []
    for filename in os.listdir(carpk_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(carpk_test_path + filename)
    test_list.sort()
    np.save('./npydata/carpk_test.npy', test_list)

    print("Generate carpk image list successfully")
except:
    print("The carpk dataset path is wrong. Please check your path.")

try:
    pucpr_train_path = pucpr_root + '/train_data/images/'
    pucpr_test_path = pucpr_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(pucpr_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(pucpr_train_path + filename)
    train_list.sort()
    np.save('./npydata/pucpr_train.npy', train_list)

    test_list = []
    for filename in os.listdir(pucpr_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(pucpr_test_path + filename)
    test_list.sort()
    np.save('./npydata/pucpr_test.npy', test_list)

    print("Generate pucpr image list successfully")
except:
    print("The pucpr dataset path is wrong. Please check your path.")

try:
    large_train_path = large_root + '/train_data/images/'
    large_val_path = large_root + '/val_data/images/'
    large_test_path = large_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(large_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(large_train_path + filename)
    train_list.sort()
    np.save('./npydata/large_train.npy', train_list)

    val_list = []
    for filename in os.listdir(large_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(large_val_path + filename)
    val_list.sort()
    np.save('./npydata/large_val.npy', val_list)

    test_list = []
    for filename in os.listdir(large_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(large_test_path + filename)
    test_list.sort()
    np.save('./npydata/large_test.npy', test_list)

    print("Generate large image list successfully")
except:
    print("The large dataset path is wrong. Please check your path.")
    
try:
    small_train_path = small_root + '/train_data/images/'
    small_val_path = small_root + '/val_data/images/'
    small_test_path = small_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(small_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(small_train_path + filename)
    train_list.sort()
    np.save('./npydata/small_train.npy', train_list)

    val_list = []
    for filename in os.listdir(small_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(small_val_path + filename)
    val_list.sort()
    np.save('./npydata/small_val.npy', val_list)

    test_list = []
    for filename in os.listdir(small_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(small_test_path + filename)
    test_list.sort()
    np.save('./npydata/small_test.npy', test_list)

    print("Generate small image list successfully")
except:
    print("The small dataset path is wrong. Please check your path.")

try:

    building_train_path = building_root + '/train_data/images/'
    building_val_path = building_root + '/val_data/images/'
    building_test_path = building_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(building_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(building_train_path + filename)
    train_list.sort()
    np.save('./npydata/building_train.npy', train_list)

    val_list = []
    for filename in os.listdir(building_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(building_val_path + filename)
    val_list.sort()
    np.save('./npydata/building_val.npy', val_list)

    test_list = []
    for filename in os.listdir(building_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(building_test_path + filename)
    test_list.sort()
    np.save('./npydata/building_test.npy', test_list)

    print("Generate building image list successfully")
except:
    print("The building dataset path is wrong. Please check your path.")
    
try:

    ship_train_path = ship_root + '/train_data/images/'
    ship_val_path = ship_root + '/val_data/images/'
    ship_test_path = ship_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(ship_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(ship_train_path + filename)
    train_list.sort()
    np.save('./npydata/ship_train.npy', train_list)

    val_list = []
    for filename in os.listdir(ship_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(ship_val_path + filename)
    val_list.sort()
    np.save('./npydata/ship_val.npy', val_list)

    test_list = []
    for filename in os.listdir(ship_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(ship_test_path + filename)
    test_list.sort()
    np.save('./npydata/ship_test.npy', test_list)

    print("Generate ship image list successfully")
except:
    print("The ship dataset path is wrong. Please check your path.")

        