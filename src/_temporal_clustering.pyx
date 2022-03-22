#cython: boundscheck=False, wraparound=False
# distutils: language=c++
from libc.math cimport sqrt
#from libc.math cimport log
import numpy as np
cimport numpy as np #is this OK?
#from libc.stdlib cimport malloc, free

from cython.parallel cimport prange
from libcpp.list cimport list as cpplist
from libc.stdio cimport printf,stdout,fflush


#https://gist.github.com/craffel/e470421958cad33df550
#from libc.stdint cimport uint32_t
#cdef extern int __builtin_popcount(unsigned int) nogil #Count 1's in a 32-bit int
#cdef extern int __builtin_popcountll(unsigned long long) nogil


# copy declarations from libcpp.vector to allow nogil
#https://stackoverflow.com/questions/7403966/most-efficient-way-to-build-a-1-d-array-list-vector-of-unknown-length-using-cyth
#cdef extern from "<vector>" namespace "std":
#	cdef cppclass vector[T]:
#		void push_back(T&) nogil
#		size_t size()
#		T& operator[](size_t)

#cdef struct Pair:
#	np.int_t id1
#	np.int_t id2
	
#ctypedef Pair* Pairptr

#https://stackoverflow.com/questions/51425300/python-fast-cosine-distance-with-cython
#cdef double cy_cosine(np.ndarray[np.float64_t] x, np.ndarray[np.float64_t] y) nogil:
cpdef double cy_cosine(np.float32_t[:] x, np.float32_t[:] y) nogil:
	cdef double xx=0.0
	cdef double yy=0.0
	cdef double xy=0.0
	cdef Py_ssize_t i
	for i in range(x.shape[0]):
		xx+=x[i]*x[i]
		yy+=y[i]*y[i]
		xy+=x[i]*y[i]
	return 1.0-xy/sqrt(xx*yy)

#cpdef int cy_hamming(long long x, long long y) nogil:
#	return __builtin_popcountll(x^y)


def find_matches(list embeddings, float threshold, int start=0, int end=-1, bint allmatches=False):
	cdef int L=len(embeddings) #embeddings.shape[0]
	#print(L)
	#cdef list matches=[]
	#cdef vector[vector[int]] matches
	cdef cpplist[cpplist[int]] matches
	cdef cpplist[int] tmp
	cdef int idx1,idx2
	cdef double dist
	cdef double mindist
	cdef int mindistidx
	cdef long counter=0
	
	if end==-1:
		end=L
	#cdef Pair* pair
	for idx1 in range(start,end): #prange(L,nogil=True):
		embed1=embeddings[idx1]
		mindist=threshold+100
		for idx2 in range(0,idx1):
			embed2=embeddings[idx2]
			dist=cy_cosine(embed1,embed2)
			if allmatches and dist<=threshold:
				matches.push_back([idx1,idx2])
			elif dist<mindist:
				mindist=dist
				mindistidx=idx2
		if not allmatches and mindist<=threshold:
			#pair=<Pair *>malloc(sizeof(Pair))
			#pair.idx1=1#idx1
			#pair.idx2=2#mindistidx
			#print(counter)
			matches.push_back([idx1,mindistidx])
		counter+=1
		if counter%1000==0:
			printf(".")
			if counter%1000000==0:
				printf("\n")
			fflush(stdout)
	#print("returning")
	#print(matches.size())
	printf("\n")
	fflush(stdout)
	return matches
	#cdef np.ndarray[np.int_t,np.int_t] a = np.empty(matches.size(), dtype=np.int)
	#cdef np.int_t[:,:] a= np.empty(matches.size(), dtype=np.int)
	#cdef int i
	#for i in prange(a.shape[0], nogil=True):
		#a[i,0] = matches[i].id1
		#a[i,1] = matches[i].id2
		#a[i][0]=matches[i,<int>0]
		#a[i][1]=matches[i,<int>1]
	#	tmp=matches[i]
	#return a


