import os
import json
import fnmatch
from pprint import pprint

def getRegionLabelDic(dic_file_path,split_key,index):
	region_labels = []
	region_label_file = open(dic_file_path)
	line = region_label_file.readline()
	while line:
		word_list = line.split(split_key)
		region_labels.append(word_list[index])
		line = region_label_file.readline()
	region_label_file.close()
	return region_labels

def getImgLabelDic(dic_file_path,split_key,index):
	img_labels = []
	img_label_file = open(dic_file_path)
	line = img_label_file.readline()
	while line:
		word_list = line.split(split_key)
		img_labels.append(word_list[index])
		line = img_label_file.readline()
	img_label_file.close()
	return img_labels
	
def getJsonFiles(dataset_path = './'):
	matches = []
	paths = []
	for root, dirnames, filenames in os.walk(dataset_path):
		for filename in fnmatch.filter(filenames, 'via_region_data*.json'):
			matches.append(os.path.join(root, filename))
			paths.append(root)	
	return matches, paths

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area
