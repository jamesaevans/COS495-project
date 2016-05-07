# Build space-delimmed list linking images to labels

import glob, os, sys, csv
BASE_DIR = '/Users/ASun/COS495-project/'
os.chdir(BASE_DIR)

with open('driver_imgs_list.csv', 'r') as file:
    csvfile = csv.reader(file, delimiter = ',')
    next(csvfile, None) # skip the header
    for row in csvfile:
        # ommitted drivers chosen randomly
        
        filepath = './small_imgs/train/' + row[1] + '/' + row[2]
        if sys.argv[1] == 'validate':
            if row[0] == 'p002' or row[0] == 'p052' or row[0] == 'p072':
                print filepath + ' ' + row[1][-1]
        else:
            if row[0] != 'p002' and row[0] != 'p052' and row[0] != 'p072':
                print filepath + ' ' + row[1][-1]

