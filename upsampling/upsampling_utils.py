#---------------------------------------------#
# Code for upsampling			      #
#---------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import itertools
#from scipy.interpolate import interp2d, RectBivariateSpline
import math
import splines
#from helper import plot_spline_2d, plot_tangent_2d

def avg_test(matrix,pts):
	vals = [matrix[pt] for pt in pts]	
	return np.mean(vals)

def cr_spline(matrix, points1):
	#points1 = [(0,0), (4,0), (0, 4), (4, 4)]
	# endconditions is a parameter
	# alpha = 0: uniform, 0.5: centripetal, 1: chordal.
	d3_pts = []
	for point in points1:
		#import pdb; pdb.set_trace()
		d3_pts.append((point[0], point[1], matrix[point]))

	s1 = splines.CatmullRom(d3_pts, alpha = 0, endconditions='natural') # endconditions natural or closed
	#print('input points', points1)
	out = s1.evaluate(1.5)[-1]
	#print(out)
	return out

	#fig, ax = plt.subplots()
	#plot_spline_2d(s1, ax=ax)
	# in uniform, can easily interpolate. evaluate(0) gives first point
	# 1 gives second. 1.5 gives halfway between 2 and 3. etc. 
	#s1.evaluate(2) 
	# points can be in 3d. and makes it easier, because you can see the actual point you want in 
	# the first values 
	#print(s1.evaluate([0,1,2,3,4])) # time points if we keep endconditions closed
	#print(s1.evaluate(0.5))
	#print(s1.evaluate(1.5)) # point we want. 
	#print(s1.evaluate(2.5))
	#print(s1.evaluate(3.5)) # another possible one
	#print(s1.evaluate(3.2))	

def plot_matrix(matrix):
	plt.imshow(matrix, interpolation='nearest', vmin = 0, vmax = matrix.shape[0]*2)
	print(np.round(matrix, 2))
	plt.show()

# gets upper right, upper left, bottom right, bottom left points for 
# diamond and averages plus some random quantity
def get_diamond(point, dist, mat, data_range, interp_f):
	# imagine bottom left is (0,0). 
	# point[0] is x axis, point[1] is y axis
	ul = (point[0] - dist, point[1] - dist)
	ur = (point[0] - dist, point[1] + dist)
	bl = (point[0] + dist, point[1] - dist)
	br = (point[0] + dist, point[1] + dist)
	d_pts = [ul, ur, bl, br]
	#d_pts = list(map(lambda x: (int(x[0]), int(x[1])), d_pts))
	#print('point: ' + str(point) + '\n\ndiagonal points' + str(d_pts))
	#d_vals = [mat[pt] for pt in d_pts]	
	rand = np.random.randint(data_range[0], data_range[1]) # TODO: modify depending on values
	#rand = 0
	out = interp_f(mat, d_pts) + rand	# TODO: interp
	return out

# diamond step
def diamond_step(matrix, iter_num, data_range, interp_f):
	#print(iter_num)
	# get central points
	#print(matrix)
	m_w = matrix.shape[0]
	div = 2**(iter_num-1)
	first_step = int((m_w//2)/div)
	#print(first_step)
	vect = list(map(int, np.arange(first_step, m_w, 2*first_step)))
	#import pdb; pdb.set_trace()	
	pt_list = list(itertools.product(vect, vect))
	#pt_list = list(map(lambda x: (int(x[0]), int(x[1])), pt_list))
	x_v = [v[0] for v in pt_list]
	y_v = [v[1] for v in pt_list]
	#print(x_v, y_v)
	#plt.figure()
	#plt.plot(x_v,y_v, marker='.', color='k', linestyle='none')
	#plt.xlim(0, m_w-1)
	#plt.ylim(0, m_w-1)
	#plt.show()
	# get diagonally adjacent points for average
	# dist is horizontal and vertical distance 
	new_mat = deepcopy(matrix)
	dist = first_step
	for point in pt_list:
		#print('point ', point)
		#import pdb; pdb.set_trace()
		new_mat[point] = get_diamond(point, dist, new_mat, data_range, interp_f)
	
	return np.array(new_mat), pt_list, int(dist)

def get_square(point, dist, size):
	up = (point[0] - dist, point[1])
	bp = (point[0] + dist, point[1])
	lp = (point[0],  point[1] - dist)
	rp = (point[0], point[1] + dist)
	p_list = [lp, up, bp, rp]
	#p_list = list(map(lambda x: (int(x[0]), int(x[1])), p_list))
	out_pts = []	
	# remove point if out of frame
	for p in p_list:
		if (p[0] >= 0 and p[0] < size) and (p[1] >=0 and p[1] < size):
			out_pts.append(p)
	return out_pts

# square step
def square_step(matrix, pt_list, dist, data_range, interp_f):
	m_copy = deepcopy(matrix)
	# depends on last diamond step. given the points and dist its easy
	#import pdb; pdb.set_trace()
	for point in pt_list:
		# get square of points
		new_pts = get_square(point, dist, matrix.shape[0])
		# get square of points around each of the square points
		#print(point)
		#print(new_pts)
		for new_pt in new_pts:
			pts_to_avg = get_square(new_pt, dist, matrix.shape[0])
			# average those points to get the new value			
			rand = np.random.randint(data_range[0], data_range[1]) # TODO: modify depending on values
			#rand = 0
			
			m_copy[new_pt] = interp_f(matrix, pts_to_avg) + rand # TODO: splines
	return m_copy
	
def diamond_square_algorithm(input, interp_f, temp):
	# iterate through sets of 4 points (overlapping windows)
	#data_range = [np.min(input), np.max(input)]
	#data_range = [-1, 1]
	#data_range = [-10, 10]
	#data_range = [-5, 5]
	#data_range = [-20, 20]
	#data_range = [-50, 50]
	#data_range = [-75, 75]
	#data_range = [-100, 100]
	#data_range = [0, 1]
	data_range = [-temp, temp]
	num_iter = int(math.ceil(math.log(input.shape[0] - 2, 2)))
	num_iter = np.maximum(1, num_iter)
	for i in range(num_iter):
		#print(i)
		if i == 0: mat = input
		#import pdb; pdb.set_trace()
		mat, pts, d = diamond_step(mat, i+1, data_range, interp_f)
		#print(d, pts)
		#plot_matrix(mat)
		#print(mat)
		mat = square_step(mat, pts, d, data_range, interp_f)
		#plot_matrix(mat)
		#print(mat)
	return mat

def bilinear(mat):
	size = mat.shape[0]
	new_mat = deepcopy(mat)
	ur = (0, size-1)
	br = (size-1, size-1)
	ul = (0, 0)
	bl = (size-1, 0)
	corner_pts = [ul, ur, bl, br]
	for i in range(size):
		for j in range(size):
			if (i,j) in corner_pts:
				continue
			else:
				#get x axis (j)
				ux = (mat[ur]*(j) + mat[ul]*(size-1-j))/(size-1)
				bx = (mat[br]*(j) + mat[bl]*(size-1-j))/(size-1)
				# get y axis (i)
				out = (ux*(size-1-i) + bx*(i))/(size-1)
				#print(out)
				new_mat[(i,j)] = out		
				
	return new_mat

def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

def bicubic(mat, h, a):
    hi,wi= mat.shape
    print(hi,wi)
    dst = np.zeros((hi,wi))
    for i in range(wi):
        for j in range(hi):
                x, y = i * h + 2 , j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
                mat_m = np.matrix([[mat[int(y-y1),int(x-x1)],mat[int(y-y2),int(x-x1)],mat[int(y+y3),int(x-x1)],mat[int(y+y4),int(x-x1)]],
                                   [mat[int(y-y1),int(x-x2)],mat[int(y-y2),int(x-x2)],mat[int(y+y3),int(x-x2)],mat[int(y+y4),int(x-x2)]],
                                   [mat[int(y-y1),int(x+x3)],mat[int(y-y2),int(x+x3)],mat[int(y+y3),int(x+x3)],mat[int(y+y4),int(x+x3)]],
                                   [mat[int(y-y1),int(x+x4)],mat[int(y-y2),int(x+x4)],mat[int(y+y3),int(x+x4)],mat[int(y+y4),int(x+x4)]]])
                mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
                dst[j, i] = np.dot(np.dot(mat_l, mat_m),mat_r)

    return dst
    

if __name__ == '__main__':
	# 4 points can be upsampled to 3x3, then 5x5, then 9x9, then 17
	# so 3*2 -1 = 5, 5*2 - 1 = 9, 9*2 -1 = 17 
	#mat = np.zeros((65,65))
	mat = np.zeros((129,129))
	#mat = np.zeros((257, 257))
	#mat = np.zeros((33,33))
	#mat = np.zeros((9,9))
	mat[0][0] = 5
	mat[0][-1] = 2
	mat[-1][0] = 1
	mat[-1][-1] = 3
	#plot_matrix(mat)
	#import pdb; pdb.set_trace()
	temp = 100
	#mat = diamond_square_algorithm(mat, avg_test, temp)
	
	mat = diamond_square_algorithm(mat, cr_spline, temp)
	
	#mat = bilinear(mat)
	#mat = bilinear2(mat)
	plot_matrix(mat)
	#cr_spline(0)
	'''
	mat, pts, d = diamond_step(mat, 1, [-1, 1])
	plot_matrix(mat)
	mat = square_step(mat, pts, d)
	plot_matrix(mat)
	mat, pts, d = diamond_step(mat, 2, [-1, 1])	
	plot_matrix(mat)
	mat = square_step(mat, pts, d)
	plot_matrix(mat)
	mat, pts, d = diamond_step(mat, 3, [-1, 1])
	plot_matrix(mat)
	mat = square_step(mat, pts, d)
	plot_matrix(mat)
	#diamond_step(mat, 4, [-1, 1])
	#diamond_step(mat, 5, [-1, 1])
	'''
