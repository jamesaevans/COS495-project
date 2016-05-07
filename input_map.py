# Build space-delimmed list linking images to labels

import glob, os, sys

# Change this according to location of project
BASE_DIR = '/Users/mjwattendorf/COS495-project'
os.chdir(BASE_DIR)

if len(sys.argv) != 2:
	print "Usage: python input_map.py <directory> > output.csv"
elif sys.argv[1] == 'test':
	folder = sys.argv[1]
	for file in glob.glob("./imgs/" + folder + "/*.jpg"):
		# Kind of hacky, but set a fake class for ease of input to tensorflow
		print file + " " + str(0)
else:
	# Iterate over a training folder, linking images to labels (ints)
	folder = sys.argv[1]
	for i in range(0, 10):
		for file in glob.glob("./imgs/" + folder + "/c" + str(i) + "/*.jpg"):
			print file + " " + str(i)

