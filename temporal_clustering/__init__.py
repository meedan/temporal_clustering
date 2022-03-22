from ._temporal_clustering import find_matches, cy_cosine
import numpy

def temporal_cluster(embeddings,threshold):

	if isinstance(embeddings,numpy.ndarray):
		embeddings=[x for x in embeddings]

	links=find_matches(embeddings,threshold)


	clusters=[x for x in range(len(embeddings))] #numbered 0 to n-1
	for a,b in links: #not sure this works :-) 
		assert a>b, "First index in pair should always be greater than second"
		clusters[a]=clusters[b] #Put a in the same cluster as b

	clean_clusters=[-1]*len(embeddings)
	unqvals=[]
	swaps={}
	next=0
	for i,c in enumerate(clusters):
		if not c in unqvals:
			unqvals.append(c)
			swaps[c]=next
			next+=1
		clean_clusters[i]=swaps[c]
	
	return clean_clusters
