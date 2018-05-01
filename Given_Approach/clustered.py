# clustering dataset
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
 

print 'Loading Data !'
mat = scipy.io.loadmat('../../data/ORL.mat')
print 'Data Loaded !'

# print mat
print mat.keys()
print 'No. of samples:',len(mat['X'])
print 'No. of Labels:',len(mat['Y'])
print mat['X'].shape
print mat['Y'].shape

m=0
for i in mat['X']:
	if max(i)>m:
		m=max(i)

plt.plot()
plt.xlim([0, 300])
plt.ylim([0, 300])
plt.title('Dataset')
for i in xrange(0,len(mat['X'])-1):
	plt.scatter(mat['X'][i],mat['X'][i+1])
plt.show()

# x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
# x2 = np.array([5, 4, 6, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
 
# plt.plot()
# plt.xlim([0, 300])
# plt.ylim([0, 300])
# plt.title('Dataset')
# # plt.scatter(x1, x2)
# plt.show()

plt.plot()
plt.xlim([0, 300])
plt.ylim([0, 300])
plt.title('Dataset')
for i in xrange(0,len(mat['X'])-200):
	plt.scatter(mat['X'][i][:10],mat['X'][i+1][:10])
plt.show()

 
# X = []
# for i in mat['X']:
# 	X.append(i)

# # # create new plot and data
# # plt.plot()
# # X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
# colors = ['b', 'g', 'r']
# markers = ['o', 'v', 's']
 
# # KMeans algorithm 
# K = 3
# kmeans_model = KMeans(n_clusters=K).fit(X)
 
# # plt.plot()
# c=0
# for i, l in enumerate(kmeans_model.labels_):
# 	try:
# 		c+=1
# 		# plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
# 		plt.plot(mat['X'][i], mat['X'][i+1], color=colors[l], marker=markers[l],ls='None')
# 		plt.xlim([0, 10])
# 		plt.ylim([0, 10])
# 		if i == 399:
# 			break
# 	except:
# 		break
# print c
 
# plt.show()
