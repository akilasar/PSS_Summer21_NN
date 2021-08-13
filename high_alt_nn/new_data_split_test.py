#-----------------------------------------------------------------------#
# Author: Eric Ham                                                      #
# Date Updated: 11/17/20                                                #
# Description: Experiments to divide up points with clustering/graph    #
# partitioning for better val/train distributions.                      #
#-----------------------------------------------------------------------#
import numpy as np
import itertools
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# use kmeans++ to get centroids that are much more spread out distance-wise to
# avoid overlapping. ie, less similar (lower overlap in our case)
def get_centroids(points, similarity_f, k = 2, *extra_args):
    centroids = []
    # draw random point for first centroid
    centroids.append(points[np.random.randint(points.shape[0]),:])
    # need k centroids. have 1, so go have to find k-1 now
    # 1 in our case, but general here for illustration of algorithm
    for cid in range(k-1):
        max_sim_list = [] # stores the max similarity over centroids centroid for a
        # given point. use this to determine which point should be next centroid
        # won't select same point for centroid twice because there is guaranteed to be a
        # less similar point
        # go through points in list 
        for i in range(points.shape[0]):
            point = points[i,:]
            max_sim = 0
            # compare point with centroids. store the highest similarity to
            # current list as lower bound on performance wrt set of centroids.
            # could also do average perhaps? in our case, just 1 centroid, so
            # 1 point
            for j in range(len(centroids)):
                #import pdb; pdb.set_trace()
                cent_sim = similarity_f(point, centroids[j], *extra_args)
                # find maximum of similarities between point and centroids. 
                # then use this as lower bound for this points performance as a
                # new centroid
                max_sim = max(max_sim, cent_sim)
            # save worst case similarity (most similar) for this point over
            # centroids
            max_sim_list.append(max_sim)

        # select data point with minimum similarity for next centroid
        # find the point with minimum worst case similarity (minimum max
        # similarity over centroids)
        next_centroid = points[np.argmin(max_sim_list),:]
        centroids.append(next_centroid)

    return centroids
def get_rand_centroids(points,k=2):
    from numpy.random import default_rng
    rng = default_rng()
    point_is = rng.choice(len(points), 2, replace=False)
    return points[point_is]

def fs_kmeans_init(points,similarity_f, cluster0_p, centroid_init='kmeans++', *extra_args):
    # get 2 centroids
    if centroid_init == 'random':
        # draw 2 random points without replacement
        centroids = get_rand_centroids(points, 2)
        print('random')

    elif centroid_init == 'kmeans++':
        centroids = get_centroids(points, similarity_f, 2, *extra_args)
    else:
        print('unsupported init type')

    # order points by relative similarity between centroid1 and centroid2
    point_dict = {}
    #import pdb; pdb.set_trace()
    for point in points:
        s0 = similarity_f(point, centroids[0], *extra_args)
        s1 = similarity_f(point, centroids[1], *extra_args)
        point_dict[tuple(point)] = s1-s0 # this makes it hashable
    #import pdb; pdb.set_trace()    
    # sort dict. larger value means more similar to centroids[1] than
    # centroids[0]
    # this will order least to greatest, so closer to centroid[0] than to
    # centroid[1] to closer to centroid[1] than to centroid[0]
    # item[0] is the "key" which is the distance
    sorted_point_dict = {k:v for k, v in sorted(point_dict.items(), key =lambda item: item[1])}
    
    # put points in clusters based on order of preference
    cluster0 = []
    cluster1 = []
    
    # remove centroids from dict so no duplicate
    del sorted_point_dict[tuple(centroids[0])]
    del sorted_point_dict[tuple(centroids[1])]
    
    cluster0.append(centroids[0])
    cluster1.append(centroids[1])
    
    # add to cluster 0 (val, smaller) first
    num_cluster0 = cluster0_p*len(sorted_point_dict)
    count = 0
    for k, v in sorted_point_dict.items():
        #print(k)
        if count >= num_cluster0:
            cluster1.append(k)
        else:
            count+=1
            cluster0.append(k)
    # return the two clusters
    return cluster0,centroids[0],  cluster1, centroids[1]

# transfer
def perform_transfer(c0_switch_cands, c1_switch_cands):
    #  sort points
    
    sort_c0_sw_cands = {k:v for k, v in sorted(c0_switch_cands.items(), key =lambda item: item[1])}
    sort_c1_sw_cands = {k:v for k, v in sorted(c1_switch_cands.items(), key =lambda item: item[1])}
    # if negative, it wants to swap
    # go through list from most negative up. if swap is overall beneficial,
    # perform it, or if no swap necessary (can swap and stay within the margin,
    # then do so). 
    # beneficial: if new member overlap w/ centroid > old member and at least same for
    # other, def. 
    # if most negative possible swap would not benefit, then no reason to
    # consider others, since these are the ones that would bring about a
    # greatest change --> exit.
   
    switch_count = 0 # --> keep track of switches
    c0_list = list(sort_c0_sw_cands.keys())
    c1_list = list(sort_c1_sw_cands.keys())
    c0_v_list = list(sort_c0_sw_cands.values())
    c1_v_list = list(sort_c1_sw_cands.values())
    
    #import pdb; pdb.set_trace() 
    # go through points in val set first
    i = 0 
    while i < len(c0_list) and i < len(c1_list):

        # if point should switch clusters
        # obvious switch (mutually beneficial or not harmful to one but
        # beneficial to other)
        if c0_v_list[i] < 0 and c1_v_list[i] <= 0:
            switch_count+=1
        # if definitely beneficial to one, but not to other, check if harm
        # caused is less in magnitude than benefit
        elif c0_v_list[i] < 0 and c1_v_list[i] > 0:
            if np.abs(c0_v_list[i]) > np.abs(c1_v_list[switch_count]):
                switch_count+=1
            else:
                break
        else: 
            break

        i += 1

    print('num switches: ', switch_count)
    # perform transfer of keys
    c0_keys_new1 = []
    c0_vals_new1 = []
    # get unswitched points
    c0_keys_new1.extend(c0_list[switch_count:])
    c0_vals_new1.extend(c0_v_list[switch_count:])
    #get points from c1
    c0_keys_new1.extend(c1_list[:switch_count])
    # flip sign on value because switching cluster
    c0_vals_new1.extend(list(map(lambda x: -x, c1_v_list[:switch_count])))
    
    c1_keys_new1 = []
    c1_vals_new1 = []
    # get points from c0
    c1_keys_new1.extend(c0_list[:switch_count])
    c1_vals_new1.extend(list(map(lambda x: -x, c0_v_list[:switch_count])))
    # get points that haven't  switched. 
    c1_keys_new1.extend(c1_list[switch_count:])
    c1_vals_new1.extend(c1_v_list[switch_count:])

    return dict(zip(c0_keys_new1, c0_vals_new1)), dict(zip(c1_keys_new1,c1_vals_new1))

# iteration step for fixed size k means
def fs_k_means_iter(clust0, clust1, sim_f, *extra_args):
    # compute cluster means (central point)
    
    #import pdb; pdb.set_trace()
    new_c0 = np.mean(clust0, axis = 0)
    new_c1 = np.mean(clust1, axis = 0)
    # plots normal here
    #plot_in_3d(clust0, new_c0, clust1, new_c1)
    # sort points in cluster by distance difference (overlap with centroid
    # minus overlap to centroid of other cluster)
    # so want to move most (s1 >> s0 first)
    c0_switch_cands = {}
    for point in clust0:
        s0 = sim_f(point, new_c0, *extra_args)
        s1 = sim_f(point, new_c1, *extra_args)
        c0_switch_cands[tuple(point)] = s0 - s1 

    c1_switch_cands = {}
    for point in clust1:
        s0 = sim_f(point, new_c0, *extra_args)
        s1 = sim_f(point, new_c1, *extra_args)
        #when sort, points where s0 >> s1 (points that should switch to c0)  will be at front
        c1_switch_cands[tuple(point)] = s1 - s0 
    
    # seems ok through here
    #import pdb; pdb.set_trace()
    # c0 switches first

    #import pdb; pdb.set_trace()
    new_c0_dict, new_c1_dict = perform_transfer(c0_switch_cands, c1_switch_cands)
    #plot_in_3d(list(new_c0_dict.keys()), new_c0, list(new_c1_dict.keys()), new_c1)
    # c1 switches second
    # switch the outputs as well. 
    c1_final_dict, c0_final_dict = perform_transfer(new_c1_dict, new_c0_dict)
    #import pdb; pdb.set_trace()
    c0_final = list(c0_final_dict.keys())
    c1_final = list(c1_final_dict.keys())
    #plot_in_3d(c0_final, new_c0, c1_final, new_c1)
    # not sure if should switch highest priority switch ones first. 
    # you should also switch if improves even if not negative. 
    # if both negative, switch. if one negative, one positive, if magnitude of
    # negative > magnitude of positive, make switch. 
    # you might want to attach a weight to this. 

    # return resulting clusters and notification as to swaps done. 
    return c0_final, new_c0, c1_final, new_c1

# fixed size k means
# function for clustering data into 2 clusters (train and validation) of a
# desired size. # points: points to cluster
# margin: allows for size variability, which defines how far
# the clusters can deviate from their requested sizes.
# cluster0_p: percentage of points that should be in cluster 1
# tol: tolerance: defines limit on what is considered a beneficial swap
# max_iter: defines how long the algorithm will go for in the worst case
# dist_f: distance function
 
def fs_kmeans(points, tol, max_iter, sim_f, margin = 0.02, cluster0_p = 0.2, init_type='kmeans++', *extra_args):
    total_s = len(points)
    margin_s = margin*total_s
    cluster0_s = cluster0_p*total_s
    cluster1_s = total_s - cluster0_s
    #clust0_range = [clust0_s - margin_s, clust0_s + margin_s] 
    #c0, cent0, c1, cent1 = fs_kmeans_init(box_pts,get_overlapf, cluster0_p, 'kmeans++', w, l)
    clust0, cent0, clust1, cent1 = fs_kmeans_init(points, sim_f, cluster0_p, init_type, *extra_args)
    # stopping criteria:
    # 1) max itr is limit of iterationss
    # 2) centroids don't change
    prev_cent0, prev_cent1 = cent0, cent1
    cent_change = sys.maxsize
    itr = 0
    
    
    # points normal here
    plot_in_3d(clust0, cent0, clust1, cent1)
    #import pdb; pdb.set_trace()
    while itr < max_iter and cent_change > tol:
        clust0, cent0, clust1, cent1 = fs_k_means_iter(clust0, clust1, sim_f, *extra_args)
        #import pdb; pdb.set_trace()
        cent_diff0, cent_diff1 = np.sum((prev_cent0-cent0)**2), np.sum((prev_cent1-cent1)**2)
        cent_change = np.mean([cent_diff0, cent_diff1])
        prev_cent0 = cent0
        prev_cent1 = cent1
        print(cent_change)
        itr+=1
    #import pdb; pdb.set_trace()
    #plot_in_3d(clust0, cent0, clust1, cent1)
    return clust0, cent0, clust1, cent1

# function for getting the overlap between two points. 
# note: overlap only occurs for samples on same z plane. 
# w = height of sample (y), l = length of sample (x)
# point format: [y, x, z] as y gives you the row in the image (dim 1)
# basically just the overlap on x,y axes
def get_overlap(point1, point2, w, l):
    # if in diff plane, overlap = 0
    if point1[-1] != point2[-1]:
        overlap = 0
    else:
        overlap_factor = 1
        # max possible overlap on y axis is w, max possible on x is l
        y_dist = np.abs(point1[0] - point2[0])
        y_overlap = np.maximum(0, w - y_dist)
    
        x_dist = np.abs(point1[1] - point2[1])
        x_overlap = np.maximum(0, l - x_dist)
        overlap = x_overlap*y_overlap*overlap_factor
    
    return overlap

# this takes z into account (z overlap) but with weighting based on distance
def get_overlap2(point1, point2, w, l):
    # if in diff plane, overlap = 0
    if point1[-1] != point2[-1]:
        overlap_factor = np.minimum(point1[-1], point2[-1])/np.maximum(point1[-1],point2[-1])
    else:
        overlap_factor = 1
    # max possible overlap on y axis is w, max possible on x is l
    y_dist = np.abs(point1[0] - point2[0])
    y_overlap = np.maximum(0, w - y_dist)

    x_dist = np.abs(point1[1] - point2[1])
    x_overlap = np.maximum(0, l - x_dist)
    overlap = x_overlap*y_overlap*overlap_factor
    
    return overlap

# overlap func for algorithm in which xy overlap is same regardless of z
# locations
def get_overlapf(point1, point2, w, l):
    if point1[-1] != point2[-1]:
        #overlap_factor = np.minimum(point1[-1], point2[-1])/np.maximum(point1[-1],point2[-1])
        overlap_factor = 1
    else:
        overlap_factor = 1
    # max possible overlap on y axis is w, max possible on x is l
    y_dist = np.abs(point1[0] - point2[0])
    y_overlap = np.maximum(0, w - y_dist)

    x_dist = np.abs(point1[1] - point2[1])
    x_overlap = np.maximum(0, l - x_dist)
    overlap = x_overlap*y_overlap*overlap_factor
    
    return overlap

# use sinusoidal weight to give periodically less weight to spread out weights. 
# sinusoid is shifted upward so the minimum value is 0
def get_overlap_sinx(point1, point2, w, l, l_axis):
    #import pdb; pdb.set_trace()
    # construct sinusoid
    sine_in = np.arange(0, l_axis)
    #import pdb; pdb.set_trace()
    sin_w = 5*np.sin(20*sine_in/(2*np.pi))
    mod = (sine_in-l_axis/2)**2
    #exp = np.exp(list(np.arange(0, 1, 1/l_axis)))
    #sin_e = sin_w*exp
    sin_e = sin_w*mod
    #import pdb; pdb.set_trace()
    print(point1, point2)
    weight = 1 - (sin_e[int(point1[1])] - sin_e[int(point2[1])]) # every period will have 
    # weight 1

    # max possible overlap on y axis is w, max possible on x is l
    y_dist = np.abs(point1[0] - point2[0])
    y_overlap = np.maximum(0, w - y_dist)

    x_dist = np.abs(point1[1] - point2[1])
    x_overlap = np.maximum(0, l - x_dist)
    overlap = x_overlap*y_overlap*weight
    
    return overlap

def get_overlap_sinxy(point1, point2, w, l, w_axis, l_axis):
    #import pdb; pdb.set_trace()
    # construct sinusoid
    x_sin_in = np.arange(0, l_axis)
    #import pdb; pdb.set_trace()
    x_sin = 5*np.sin(20*x_sin_in/(2*np.pi))
    x_mod = (x_sin_in-l_axis/2)**2
    #exp = np.exp(list(np.arange(0, 1, 1/l_axis)))
    #sin_e = sin_w*exp
    sin_ex = x_sin*x_mod
    #import pdb; pdb.set_trace()
    #print(point1, point2)
    x_weight = 1 - (sin_ex[int(point1[1])] - sin_ex[int(point2[1])]) # every period will have 
    # weight 1
    
    y_sin_in = np.arange(0, w_axis)
    #import pdb; pdb.set_trace()
    y_sin = 5*np.sin(20*y_sin_in/(2*np.pi))
    y_mod = (y_sin_in-w_axis/2)**2
    #exp = np.exp(list(np.arange(0, 1, 1/l_axis)))
    #sin_e = sin_w*exp
    sin_ey = y_sin*y_mod
    #import pdb; pdb.set_trace()
    #print(point1, point2)
    y_weight = 1 - (sin_ey[int(point1[0])] - sin_ey[int(point2[0])]) # every period will have 

    # max possible overlap on y axis is w, max possible on x is l
    y_dist = np.abs(point1[0] - point2[0])
    y_overlap = np.maximum(0, w - y_dist)

    x_dist = np.abs(point1[1] - point2[1])
    x_overlap = np.maximum(0, l - x_dist)
    overlap = x_overlap*y_overlap*x_weight*y_weight
    
    return overlap

#
# 
def convert_to_overlap_vect(points, overlap_func):
    # include self because this will make vectors consistent
    format_pts = []
    for point1 in points:
        point1to2 = []
        for point2 in points:
            point1to2.append(overlap_func(point1, point2, 21, 21))
        format_pts.append(np.array(point1to2))
    #import pdb; pdb.set_trace()
    return format_pts

def plot_in_xy(clust0,  cent0, clust1,cent1):
    clust0_x = []
    clust0_y = []
    clust1_x = []
    clust1_y = []
    for point in clust0:
        clust0_x.append(point[1])
        clust0_y.append(point[0])

    for point in clust1:
        clust1_x.append(point[1])
        clust1_y.append(point[0])

    plt.figure()
    plt.scatter(clust0_x, clust0_y, c = 'blue', label='cluster0 (val)')
    plt.scatter(cent0[1], cent0[0], c = 'blue', marker='*', s=100)
    plt.scatter(clust1_x, clust1_y, c = 'red', label='cluster1 (train)')
    plt.scatter(cent1[1], cent1[0], c= 'red', marker='*', s= 100)
    plt.legend()
    plt.show()

def plot_in_3d(clust0, cent0, clust1, cent1):
    clust0_x = []
    clust0_y = []
    clust0_z = []

    clust1_x = []
    clust1_y = []
    clust1_z = []

    for point in clust0:
        clust0_x.append(point[1])
        clust0_y.append(point[0])
        clust0_z.append(point[2])

    for point in clust1:
        clust1_x.append(point[1])
        clust1_y.append(point[0])
        clust1_z.append(point[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(clust0_x, clust0_y, clust0_z, c = 'blue', label='cluster0 (val)')
    ax.scatter(cent0[1], cent0[0], cent0[2], c = 'blue', marker='*', s=100)
    ax.scatter(clust1_x, clust1_y,clust1_z, c = 'red', label='cluster1 (train)')
    ax.scatter(cent1[1], cent1[0], cent1[2], c = 'red', marker='*', s= 100)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # point format!!
    # [y, x, z]
    # generate box of points
    # sample box: x = 512, y = 256, z = 10
    # simplified for test 
    #import pdb; pdb.set_trace()
    y_axis =20 
    x_axis = 20
    z_axis = 6
    box_pts = np.array(list(itertools.product(np.arange(0,y_axis),
        np.arange(0,x_axis),np.arange(1,z_axis))))
    #import pdb; pdb.set_trace()
    #box_pts = np.array(list(itertools.product(np.arange(0,8), np.arange(0,16))))
    # centroid aquisition test, verifies k++ produces lower overlaps (less
    # similar)
    w = 5
    l = 5
    '''
    rands = []
    k_plus = []
    for i in range(1000):
        points_r = get_rand_centroids(box_pts, 2)
        rands.append(get_overlapf(points_r[0], points_r[1], w, l))
    
        points_k = get_centroids(box_pts, get_overlapf, 2, w, l)
        k_plus.append(get_overlapf(points_k[0], points_k[1], w, l))

    # random init
    print('rand mean: ', np.mean(rands))
    print('rand std: ', np.std(rands))
    print('rand max: ', np.max(rands))
    print('rand min: ', np.min(rands))

    # k++ init (in general gives lower overlap --> what we want)
    print('k++ mean: ', np.mean(k_plus))
    print('k++ std: ', np.std(k_plus))
    print('k++ max: ', np.max(rands))
    print('k++ min: ', np.min(rands))
    
    # true
    true_dist = []
    for point1 in box_pts:
        for point2 in box_pts:
            true_dist.append(get_overlapf(point1, point2, w, l))
    print('All points comparison')
    print('mean: ', np.mean(true_dist))
    print('std: ', np.std(true_dist))
    print('max: ', np.max(true_dist))
    print('min: ', np.min(true_dist))
    # test init
    cluster1_p = 0.2
    #import pdb; pdb.set_trace()
    c0, cent0, c1, cent1 = fs_kmeans_init(box_pts,get_overlapf, cluster1_p, 'kmeans++', w, l)
    # rand def way worse
    #c0, cent0, c1, cent1 = fs_kmeans_init(box_pts,get_overlapf,cluster1_p,'random', w, l)
    print(len(box_pts), len(c0), len(c1)) 
    #plot_in_xy(c0, cent0, c1, cent1)
    plot_in_3d(c0, cent0, c1, cent1)
    
    '''
    # test kmeans iter
    clust0, cent0, clust1, cent1 = fs_kmeans(box_pts, 0.005, 100,
            get_overlap_sinx, 0.02, 0.2,'kmeans++', w, l, x_axis)
    # results: get_overlap --> all low (because all in diff z planes)
    #          get_overlap2 --> all low for same reason above
    #          get_overlapf --> better bc z has no impact on overlap here
    plot_in_3d(clust0, cent0, clust1, cent1)
    overlap = []
    for point1 in clust0:
        for point2 in clust1:
            overlap.append(get_overlap(point1, point2, w,l))
    print('xy overlap')
    print('mean: ', np.mean(overlap))
    print('std: ', np.std(overlap))
    print('min: ', np.min(overlap))
    print('max: ', np.max(overlap))
    #import pdb; pdb.set_trace()    
    plt.hist(overlap)
    plt.show()

    z_overlap = []
    for point1 in clust0:
        for point2 in clust1:
            z_overlap.append(get_overlap2(point1, point2, w, l))
    print('including z overlap')
    print('mean: ', np.mean(z_overlap))
    print('std: ', np.std(z_overlap))
    print('min: ', np.min(z_overlap))
    print('max: ', np.max(z_overlap))
    #
    
    '''
    # test 1:
    # k means with distance as metric, 2 clusters
    
    
    clusterer = KMeans(n_clusters=2, random_state=10, n_init = 50, max_iter=500, tol = 1e-4)
    cluster_labels = clusterer.fit_predict(box_pts)
    # evaluate overlap
    # get clusters
    clust0 = box_pts[cluster_labels==0]
    clust1 = box_pts[cluster_labels==1]
    overlap = []
    for point1 in clust0:
        for point2 in clust1:
            overlap.append(get_overlap(point1, point2, 21, 21))

    print('mean: ', np.mean(overlap))
    print('std: ', np.std(overlap))
    print('min: ', np.min(overlap))
    print('max: ', np.max(overlap))
    #import pdb; pdb.set_trace()    
    
    # plot distribution of overlap. ideally gaussian, with a manipulatable mean
    plt.hist(overlap)
    plt.show()
    # test 2:
    new_pts = convert_to_overlap_vect(box_pts, get_overlapf)
    
    clusterer = KMeans(n_clusters=2, random_state=10, n_init = 50, max_iter=500, tol = 1e-4)
    cluster_labels = clusterer.fit_predict(new_pts)
    # evaluate overlap
    # get clusters
    clust0 = box_pts[cluster_labels==0]
    clust1 = box_pts[cluster_labels==1]
    overlap = []
    for point1 in clust0:
        for point2 in clust1:
            overlap.append(get_overlap(point1, point2, 21, 21))
    print('xy overlap')
    print('mean: ', np.mean(overlap))
    print('std: ', np.std(overlap))
    print('min: ', np.min(overlap))
    print('max: ', np.max(overlap))
    #import pdb; pdb.set_trace()    
    
    z_overlap = []
    for point1 in clust0:
        for point2 in clust1:
            z_overlap.append(get_overlap2(point1, point2, 21, 21))
    print('including z overlap')
    print('mean: ', np.mean(z_overlap))
    print('std: ', np.std(z_overlap))
    print('min: ', np.min(z_overlap))
    print('max: ', np.max(z_overlap))
    #import pdb; pdb.set_trace()    

    import pdb; pdb.set_trace()
    # plot distribution of overlap. ideally gaussian, with a manipulatable mean
    plt.hist(overlap)
    plt.show()

    # evaluate the outcome
    # property 1: overlap between cluster members
    # Mean Overlap, Std overlap. Min and max overlap
    # property 2: mixing between clusters. 

    '''
