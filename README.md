# Temporal Clustering

This is a minimal example to perform online, single-link hierarchical clustering using Cython. 
You must have numpy, sentence_transformers, and Cython installed.

To install, use
```
python setup.py build
python setup.py install
```

A minimal example is in `examples/example.py`

```
$ cd examples && python example.py 

[0, 0, 1, 1, 1]
```

The output shows that the first two items form one cluster (cluster 0) and the last three items form another cluster (cluster 1)


