import sys
import os
import knn

def run(root, halt):
    for(loc,dirs,files) in os.walk(root,topdown=True):
        # print(loc)
        # print(dirs)
        # print(files)
        # print('-------')
        if(loc == (root + '\\' + halt)):
            break
        for f in files:
            if(f.lower().endswith('.jpg')):
                knn.add_to_list(loc,f)
    knn.run_knn()


#split into 5 groups of 4 years apiece?
#keep relevance in the pictures, was there a specific point in the last 20 years cameras improved?
#have argv[0] be the full uri and argv[1] be the first folder we don't want to pull from
if(__name__ == "__main__"):
    #how can I generalize this without requiring people type this out?
    root_dir = 'D:\\College\\RJI\\pics\\1999\\Fall\\Dump'
    run(root_dir, sys.argv[1])