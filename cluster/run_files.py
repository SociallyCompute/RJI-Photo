import sys
import os
import cluster

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
    for(loc,dirs,files) in os.walk(root,topdown=True):
        print(loc)
        print(dirs)
        print(files)
        print('-------')
        # if(loc == (root + '\\' + halt)):
        #     break
        for f in files:
            if(f.lower().endswith('.jpg')):
                cluster.add_to_list(loc,f)
    n_comp = input("How many n_components would you like to compress: ")
    cluster.pca_compress(int(n_comp)) #param passed is n_components compressed in PCA
    if(alg == "1" or alg == "K-Means"):
        n_clus = input("How many clusters would you like to group: ")
        cluster.run_kmeans(int(n_clus)) #param is n_clusters in kmeans
    elif(alg == "2" or alg == "KNN"):
        n_clus = input("How many clusters would you like to group: ")
        cluster.run_knn(n_clus)
    else:
        print("No functionality provided for the algorithm yet")


#split into 5 groups of 4 years apiece?
#keep relevance in the pictures, was there a specific point in the last 20 years cameras improved?
#have argv[1] be the root folder and argv[2] be the first folder we don't want to pull from
if(__name__ == "__main__"):
    alg, data, stop = print_menu()
    # print(alg)
    # print(data)
    # print(stop)
    #how can I generalize this without requiring people type this out?
    if(data == "root" or data == '1'):
        root_dir = '/mnt/md0/mysql-dump-economists/Archives/2017/Spring/Dump/'
    else:
        root_dir = data
    run(root_dir, stop, alg)
