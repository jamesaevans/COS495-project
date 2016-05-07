from PIL import Image, ImageFilter
import os, sys, glob

# Change this according to location of project
BASE_DIR = '/Users/Michael/Documents/School/4_Senior_Spring/Cos495/Final_Project/COS495-project'
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

def thresholdImages(folder):
	for i in range(0, 10):
		print i * 10, "percent done"
		for file in glob.glob("./small_imgs/" + folder + "/c" + str(i) + "/*.jpg"):
			outfile = file.replace('small_imgs', 'thresh_imgs')
			im = Image.open(file)
			gray = im.convert('L')
			bw = gray.point(lambda x: 0 if x<128 else 255, '1')
			bw.save(outfile, "JPEG")

if __name__ == '__main__':
	thresholdImages('validate')
