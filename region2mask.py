import cv2
import numpy as np
import math



def Polygon2Mask(map_size,polygon,label):
	pts = np.array(polygon)
	temp_size0 = map_size[0]
	temp_size1 = map_size[1]
	map_size[0] = temp_size1
	map_size[1] = temp_size0
	mask = np.zeros(map_size)
	cv2.fillConvexPoly(mask,pts,label)	
	#box = getBoundingBox(x,y)
	#cv2.rectangle(mask,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),label)
	#cv2.rectangle(mask,(91,91),(100,100),label)
	#cv2.namedWindow('mask',0)
	#cv2.resizeWindow('mask', 400, 300)
	#cv2.imshow('mask',mask)
	#cv2.waitKey(0)
	return mask

def Circle2Mask(map_size,circle,label):
	mask = np.zeros(map_size)
	cv2.circle(mask,(circle[0],circle[1]),circle[2],label,-1)
	return mask
	
def CaculateArea(mask):
	area = np.count_nonzero(mask)
	return area

def getPolygonCorners(x,y):
	corners = []
	count = 0
	for i in x:
		j = y[count]
		corners.append((int(i),int(j)))
		count+=1
	return corners
	
def getBoundingBox(x,y):
	x1 = min(x)
	x2 = max(x)
	y1 = min(y)
	y2 = max(y)
	box = [x1,y1,x2-x1,y2-y1]
	return box

def circle2Polygon(circle,num): #num should be num = 4*steps,steps>=1 and belong to N
	
	cx = circle[0]
	cy = circle[1]
	sx = []
	sy = []
	r = circle[2]
	
	steps = num/4
	
	radian_step = 90.0/steps
		
	#Quadrant1
	c=0
	while c < steps:
		sx.append(cx+r*math.sin(math.radians(radian_step*c)))
		sy.append(cy-r*math.cos(math.radians(radian_step*c)))
		c+=1
	#Quadrant2
	c=0
	while c < steps:
		sx.append(cx+r*math.cos(math.radians(radian_step*c)))
		sy.append(cy+r*math.sin(math.radians(radian_step*c)))
		c+=1	
	#Quadrant3
	c=0
	while c < steps:
		sx.append(cx-r*math.sin(math.radians(radian_step*c)))
		sy.append(cy+r*math.cos(math.radians(radian_step*c)))
		c+=1
	#Quadrant4
	c=0
	while c < steps:
		sx.append(cx-r*math.cos(math.radians(radian_step*c)))
		sy.append(cy-r*math.sin(math.radians(radian_step*c)))
		c+=1	
	
	return getPolygonCorners(sx,sy)
	
def corners2xy(corners):
	i=0
	l = len(corners)
	xy = []
	x = []
	y = []
	while i < l:
		xy.append(corners[i][0])
		x.append(corners[i][0])
		xy.append(corners[i][1])
		y.append(corners[i][1])
		i+=1
	return xy,x,y
	
def rect2PolygonCorners(rect):
	corners = []
	corners.append((rect[0],rect[1]))
	corners.append((rect[0]+rect[2],rect[1]))
	corners.append((rect[0]+rect[2],rect[1]+rect[3]))
	corners.append((rect[0],rect[1]+rect[3]))
	return corners
	
	
#for test

#print(circle2Polygon([263,502,92],32))

#print(CaculateArea(Polygon2Mask([1600,1200],circle2Polygon([263,502,92],32),1)))

