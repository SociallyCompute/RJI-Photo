import sys
import os
import cluster
from PIL import Image
import numpy as np

pics = {}
index_set = {}
np_pics = []
second_pics = []
reduced_pics = []
dump_files = []
edited_files = []
files = {}

def print_menu():
    print("Which algorithm would you like to train?")
    print("1: K-Means")
    print("2: KNN")
    print("3: Convolusional Neural Network")
    alg = input("Your choice: ")
    print("What dataset would you like to use?")
    print("1: root")
    print("2: Custom path")
    data = input("Your choice: ")
    print("What folder do you want to stop on?")
    stop = input("Your choice: ")
    return alg, data, stop

def run(root, halt, alg):
    # count = 0
    fi = open('labels.txt', 'a')
    for(loc,dirs,files) in os.walk('/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/',topdown=True):
        for f in files:
            if(f.lower().endswith('.jpg')):
                dump_files.append(f.lower())
    
    for(loc,dir,files) in os.walk('/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Edited/',topdown=True):
        for f in files:
            if f in dump_files:
                files[f] = 1
                fi.write(str(f) + " 1\n")
            else:
                files[f] = 0
                fi.write(str(f) + " 0\n")
                # im = Image.open(loc + '/' + f)
                # fl = open("edited22080files.txt", "a")
                # fi = open("edited18048files.txt", "a")
                # fil = open("edited9024files.txt", "a")
                # #this should never fire, it would mean there is a duplicate picture
                # if(f in pics):
                #     return
                # #add to pics dictionary with file name for key and add to np list
                # # im = im.convert('1') #convert to grayscale
                # mat2d = np.array(im)
                # print(mat2d.shape)
                # if(mat2d.ndim == 3):
                #     mat2d = mat2d.reshape((mat2d.shape[1] * mat2d.shape[2]), mat2d.shape[0])
                # if(mat2d.shape[0] == 22080):
                #     print("test")
                #     fl.write(str(loc) + "/" + str(f) + "\n")
                # elif(mat2d.shape[0] == 18048):
                #     print('test2')
                #     fi.write(str(loc) + "/" + str(f) + "\n")
                # elif(mat2d.shape[0] == 9024):
                #     print('test3')
                #     fil.write(str(loc) + "/" + str(f) + "\n")
                # else:
                #     print(mat2d.shape)
                # count = count + 1
    # f.close()
    # print(count)
    # cluster.pca_compress(int(input("How many n_components would you like to compress: "))) #param passed is n_components compressed in PCA
    # if(alg == "1" or alg == "K-Means"):
    #     cluster.run_kmeans(int(input("How many clusters would you like to group: "))) #param is n_clusters in kmeans
    # elif(alg == "2" or alg == "KNN"):
    #     cluster.run_knn(int(input("How many clusters would you like to group: "))
    # else:
    #     print("No functionality provided for the algorithm yet")

def run_files():
    count = 0
    f = open("18048files.txt", "r")
    for line in f:
        count = count + 1
        line = "../../../../.." + line.rstrip()
        print(line)
        mat3d = np.array(Image.open(line))
        mat3d = mat3d[0::2,0::2,:]
        mat2d = mat3d.reshape((mat3d.shape[0] * mat3d.shape[1]), mat3d.shape[2])
        # print(mat2d.shape)
        cluster.add_to_list_file(line, mat2d)
        if(count >= 250):
            break
    matrix = cluster.pca_file(int(input("How many components would you like to compress: ")))
    cluster.run_kmeans_file(matrix, int(input("How many clusters would you like to group: ")))


#split into 5 groups of 4 years apiece?
#keep relevance in the pictures, was there a specific point in the last 20 years cameras improved?
#have argv[1] be the root folder and argv[2] be the first folder we don't want to pull from
if(__name__ == "__main__"):
    alg, data, stop = print_menu()
    #how can I generalize this without requiring people type this out?
    if(data == "root" or data == '1'):
        # root_dir = '/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/'
        root_dir = '/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Edited/'
    else:
        root_dir = data
    
    # if alg == "1":
    #     run_files()
    # else:
    run(root_dir, stop, alg)
