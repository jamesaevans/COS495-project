# Build space-delimmed list linking images to labels

import glob, os, sys

# Change this according to location of project
BASE_DIR = '/Users/jaevans/Desktop/statefarm'
os.chdir(BASE_DIR)

if len(sys.argv) != 2:
	print "Usage: python input_map.py <directory> > output.csv"
else:
	# Iterate over a training folder, linking images to labels (ints)
	folder = sys.argv[1]
	for i in range(0, 10):
		for file in glob.glob("./imgs/" + folder + "/c" + str(i) + "/*.jpg"):
			print file + " " + str(i)

