import numpy as np
import re
import copy
import math
import json
import os
import cv2
from itertools import cycle
from sklearn.cluster import MeanShift
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from PIL import Image,ImageDraw


class way:#way of one object
    id=""#id
    size=0#number of detections
    length=0#length of way in pixels
    object_type=0#id of object type name
    frames=[]#frames of detections
    centres=[]#centres of bounding boxes
    points=[]#corners of bounding boxes
    sizes=[]#size of bounding boxes
    speeds=[]#speeds in every point
    vectorX=0#X of way's vector
    vectorY=0#Y of way's vector
    clusternum=-1#Index of cluster
    
    def __init__(self,arr,o_t):
        self.size=arr[0].size
        self.centres=np.column_stack([arr[0],arr[1]])
        self.points=np.column_stack([arr[2],arr[3]])
        self.sizes=np.column_stack([arr[4],arr[5]])
        self.frames=arr[6]
        self.object_type=o_t
        self.count_vector_length_and_speeds()
    
    def count_vector_length_and_speeds(self):
        self.speeds=[]
        if self.size>1:
            self.speeds=np.zeros(self.size)
            for i in range(self.size-1):
                buflen=math.sqrt(math.pow((self.centres[i+1][0]-self.centres[i][0]),2)+math.pow((self.centres[i+1][1]-self.centres[i][1]),2))
                if(i!=(self.size-1)):
                    self.speeds[i+1]=int(buflen/(self.frames[i+1]-self.frames[i]))
            self.vectorX=self.centres[self.size-1][0]-self.centres[0][0]
            self.vectorY=self.centres[self.size-1][1]-self.centres[0][1]
            self.length=int(math.sqrt(self.vectorX*self.vectorX+self.vectorY*self.vectorY))
    
    def printarr(self,arr):
        k=0
        rep=""
        for i in arr:
            rep +=str(i)
            if(k!=self.size-1):
                rep+=", "
            if(k==self.size-1):
                rep+=". "
            k=k+1
            if k%6==0:
                rep+="\n"
        rep+="\n"
        if k%6!=0:
            rep+="\n"
        return rep
    
    def delete_element(self,j):
        self.frames=np.delete(self.frames,j)
        self.centres=np.delete(self.centres,j,axis=0)
        self.points=np.delete(self.points,j,axis=0)
        self.sizes=np.delete(self.sizes,j,axis=0)
        self.size=self.size-1
        self.count_vector_length_and_speeds()
    
    def __str__(self):
        rep = "Size="+str(self.size)+"\n\n"
        rep += "Length=" + str(self.length) + "\n\n"
        rep += "Object type id:" + str(self.object_type) + "\n\n"
        rep += "Frames:\n"
        rep +=self.printarr(self.frames)
        rep += "Centres:\n"
        rep +=self.printarr(self.centres)
        rep += "Points:\n"
        rep +=self.printarr(self.points)
        rep+="Average square="+str(self.avgS)+ "\n"
        rep += "Sizes:\n"
        rep +=self.printarr(self.points)
        rep+="Average speed="+str(self.avgSpeed)+ "\n"
        rep += "Speeds:\n"
        rep +=self.printarr(self.speeds)
        return rep
    
    def to_save(self):
        save=[]
        save.append(int(self.object_type))
        for i in range(self.size):
            save.append([int(self.centres[i][0]),int(self.centres[i][1]),
                         int(self.points[i][0]), int(self.points[i][1]),
                         int(self.sizes[i][0]),  int(self.sizes[i][1]),
                         int(self.frames[i])])
        return save


# In[11]:


class Ways_clear(way):#stats of one cam
    size=0#num of detected objects
    ways=[]#ways of objects
    types=[]#unique_types
    W=0
    H=0
    max_id=0
    segmentated=[]
    segmentated_types=[]
    segmentated_counts=[]
    test_mode=True#test mode flag
    lines_ends=[]
    colors=[(255,0,0),#colors for _clusters
            (128,128,0),
            (0,255,0),
            (0,128,128),
            (0, 255, 255),
            (255, 255, 0),
            (0,0,255)]
    color_noize=(0,0,0)#color for noize
    
    def __init__(self,path_to_data):
        self.path_to_data=path_to_data
        self.src_name=os.path.join(path_to_data,"frame.jpg")
        self.path_to_stats=os.path.join(path_to_data,"ways_clear.txt")
        self.path_to_results=path_to_data
        self.load_ways()
        
    def load_ways(self):
        self.ways=[]
        with open(self.path_to_stats, 'r') as f:
            w = json.load(f)
            self.size=len(w)
            k=0
            for i in w:
                newway=way(np.transpose(np.array(i[1:])),i[0])
                newway.id=str(k)
                k+=1
                self.ways.append(newway)
                check=True
                for j in self.types:
                    if i[0]==j:
                        check=False
                        break
                if check==True:
                    self.types.append(i[0])
            if(self.test_mode==True):
                print("load_ways - done")

    def draw_cluster_lines(self,ways,name):
        lines=[]
        im = Image.open(self.src_name)
        draw=ImageDraw.Draw(im)
        for i in ways:
            lines.append(i.centres)
        j=0
        for line in lines:
            ind=ways[j].clusternum
            if(ind!=-1):
                color=self.colors[ways[j].clusternum]
            else:
                color=self.color_noize
            for i in range(len(line)-1):
                draw.line([(line[i][0],line[i][1]),(line[i+1][0],line[i+1][1])],fill=tuple(color),width=1)
            j=j+1
        del draw
        im.save(os.path.join(self.path_to_results,name))
    
    def load_centres_start(self):
        self.centres_start=[]
        for i in self.ways:
            self.centres_start.append([i.centres[0][0],i.centres[0][1],i.centres[i.size-1][0],i.centres[i.size-1][1]])
        self.centres_start=np.array(self.centres_start)
    
    def clustering_meanshift(self):
        self.load_centres_start()
        ms = MeanShift().fit(self.centres_start)
        labels = ms.labels_
        for i in range(self.size-1):
            self.ways[i].clusternum=labels[i]
        self.draw_cluster_lines(self.ways,"clustering_meanshift_lines.png")
        if(self.test_mode==True):
            print("clustering_meanshift - done")
        
    def __str__(self):
        rep = "Size="+str(self.size)+"\n"
        for way in self.ways:
            rep+=str(way)
        return rep
