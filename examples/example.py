# python setup.py build_ext --inplace


import datetime
from sentence_transformers import SentenceTransformer
from temporal_clustering import temporal_cluster

data=[{"datetime":datetime.datetime(2022,1,1), "text":"This is an example"},
	{"datetime":datetime.datetime(2022,1,2), "text":"This is another example"},
	{"datetime":datetime.datetime(2022,1,3), "text":"Eat an apple a day"},
	{"datetime":datetime.datetime(2022,1,4), "text":"Eat apples daily"},
]

#Step 1. Sort the data temporally


#Step 2. Embed the text

sbert=SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings=sbert.encode([x["text"] for x in data])

#Step 3. Find items that should merge given a distance threshold (lower merges less)

threshold=0.16

cluster_membership=temporal_cluster(embeddings,threshold)

print(cluster_membership)

