import os
import numpy as np
import math

#this algorithm implements fuzzy k means clustering on a set of data and produces a file with the cluster means and the members (data values) in each cluster
#modify this source code in line 92 to specify the file containing the data you want to cluster, and the number of clusters k
#run fuzzy.py in the terminal

def fuzzykmeans(data, k):
	with open(data, "rb") as f:
		d=np.loadtxt(f,delimiter="\t")
		d=d[0:,0:-1]
		
		centres= []

		centres = randomize_centres(d, centres, k)  

		old_centres = [[] for i in range(k)] 

		iterations = 0
		while not (has_converged(centres, old_centres, iterations)):
			iterations += 1

			clusters = [[] for i in range(k)]
			
			distances = distance_calc(d,centres) #calculates a distance matrix 
			w=partition(d,centres,k,distances) #calculates a partition matrix
			centres = shiftCentres(d,w) #shifts centres based on values in the partition matrix
			
		weightMatrix = np.matrix(w) #turns the partition matrix into a python matrix data structure for easier processing
		for i in range(len(d)):
			likelyCluster = np.argmax(weightMatrix[:,i]) #returns the most likely cluster that a point belongs to (ie: given a set of probabilities of belonging to various clusters, chooses the cluster with the highest probability)
			clusters[likelyCluster].append(d[i]) #adds the value to the correct cluster
			
		f=open("fuzzyoutput.txt","w")	
		print>>f,"The total number of data instances is: " + str(len(d))
		print>>f,"The total number of iterations necessary is: " + str(iterations)
		print>>f,"The means of each cluster are: " + str(centres)
		print>>f,"The clusters are as follows:"
		for cluster in clusters:
			print>>f,"Cluster with a size of " + str(len(cluster)) + " starts here:"
			print>>f,np.array(cluster).tolist()
			print>>f,"Cluster ends here."

		return
		f.close()
  
def distance_calc(d,centres):
	distances=[[0 for j in range(len(d))]for i in range(len(centres))] 
	for i in range(len(centres)):	
		for j in range(len(d)):
			distances[i][j]=(sum([math.pow(d[j][n]-centres[i][n],2)for n in range(len(d[j]))]))**(0.5) #finds the distance between each data point and each centre
			
	return distances
 
def partition(d,centres,k,distances):
	w=[[0 for j in range(len(d))]for i in range(len(centres))]
	for i in range(len(centres)):
		for j in range(len(d)):
			w[i][j]=1/sum([math.pow((distances[i][j]/distances[s][j]),2) for s in range(k)]) #calculates belongingness of a point to a centre, based on the ratio of the distance between the point and the centre relative to the other centres
	return w
 
def shiftCentres(d,w): 
	centres=[[0 for q in range(len(d[0]))]for v in range(len(w))]
	for v in range(len(w)):
		for q in range(len(d[0])):
			centres[v][q]=sum([(w[v][h]**2)*d[h][q] for h in range(len(d))])/sum([(w[v][h]**2) for h in range(len(d))]) #shifts centres based on value in partition matrix
	
	return centres
	
def randomize_centres(d, centres, k):
    for cluster in range(0, k):
        centres.append((d[np.random.randint(0, len(d), size=1)]+0.0001).flatten().tolist())
    return centres

def has_converged(centres, old_centres, iterations):
    MAX_ITERATIONS = 1000
    if iterations > MAX_ITERATIONS:
        return True
    return old_centres == centres
	
def main():
	fuzzykmeans("test_data.txt",3)
	
if __name__ == "__main__":
    main()