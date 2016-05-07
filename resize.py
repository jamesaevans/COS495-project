from PIL import Image, ImageFilter
import os, sys, glob

# Change this according to location of project
BASE_DIR = '/Users/jaevans/Desktop/COS495-project'
os.chdir(BASE_DIR)

def resizeLabeledImages(folder, size):
	for i in range(0, 10):
		instring = "./imgs/" + folder + "/c" + str(i) + "/*.jpg"
		for file in glob.glob("./imgs/" + folder + "/c" + str(i) + "/*.jpg"):
			outfile = file.replace('imgs', 'small_imgs')
			im = Image.open(file)
			im.thumbnail(size, Image.ANTIALIAS)
			im.save(outfile, "JPEG")

def resizeUnlabeledImages(folder, size):
	for i in range(0, 10):
		for file in glob.glob("./imgs/" + folder + "/*.jpg"):
			outfile = file.replace('imgs', 'small_imgs')
			im = Image.open(file)
			im.thumbnail(size, Image.ANTIALIAS)
			im.save(outfile, "JPEG")

def findEdges(folder):
	for i in range(0, 10):
		for file in glob.glob("./small_imgs/" + folder + "/c" + str(i) + "/*.jpg"):
			outfile = file.replace('small_imgs', 'edge_imgs')
			im = Image.open(file)
			im = im.filter(ImageFilter.FIND_EDGES)
			im.save(outfile, "JPEG")

def findEdgesUnlabeled(folder):
	for i in range(0, 10):
		for file in glob.glob("./small_imgs/" + folder + "/*.jpg"):
			outfile = file.replace('small_imgs', 'edge_imgs')
			im = Image.open(file)
			im = im.filter(ImageFilter.FIND_EDGES)
			im.save(outfile, "JPEG")

if __name__ == '__main__':
	findEdgesUnlabeled('test')
