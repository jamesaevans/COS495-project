# Convert probability assignments to a (possibly weakly) hard
# clustering with k non-zero assignments
def convertSoftClustering(k, probs):
	sortedProbs = sorted(probs, reverse=True)

	# Sum of k largest assignments
	largSum = sum(sortedProbs[0:k])

	for i in range(len(probs)):
		if sortedProbs.index(probs[i]) < k:
			probs[i] = probs[i]/largSum
		else:
			probs[i] = 0

	return probs

# Convert a submission file of soft clusterings to k-hard clustering
def convertClusteringFile(filename, k):
	f = open(filename)
	clusterings = f.readlines()
	for i in range(1,len(clusterings)):
		line = clusterings[i]
		probs = line.split(',')[1:]
		filename = line.split(',')[0]

		nums = []
		for p in probs:
			nums.append(float(p))
		converted = convertSoftClustering(k,nums)
		a = "%s,%0.04f,%0.04f,%0.04f,%0.04f,%0.04f,%0.04f,%0.04f,%0.04f,%0.04f,%0.04f" % (filename,converted[0], converted[1], converted[2], converted[3], converted[4], converted[5], converted[6], converted[7], converted[8], converted[9])
		print a

if __name__ == '__main__':
	convertClusteringFile('submission-1.txt', 1)