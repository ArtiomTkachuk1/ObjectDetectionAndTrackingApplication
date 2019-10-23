import numpy as np
import pandas as pd
import re
import copy
import math
import json
import bezier
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw,ImageFont
from scipy.optimize import curve_fit
from sklearn.cluster import MeanShift

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
    lengths=[]#length of way in every point
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
            self.lengths=np.zeros(self.size)
            for i in range(self.size-1):
                if(i!=(self.size-1)):
                    buflen=math.sqrt(math.pow((self.centres[i+1][0]-self.centres[i][0]),2)
                                     +math.pow((self.centres[i+1][1]-self.centres[i][1]),2))
                    self.speeds[i+1]=int(buflen/(self.frames[i+1]-self.frames[i]))
                    #it's better than count length between every
                    vX=self.centres[i+1][0]-self.centres[0][0]
                    vY=self.centres[i+1][1]-self.centres[0][1]
                    self.lengths[i+1]=int(math.sqrt(vX*vX+vY*vY))
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
class heatmap():
    s=0#Size of area in heatmap
    W=0#Number of areas in heatmap's row
    H=0#Number of areas in heatmap's
    name=""#name of resulting image
    src_name=""#src image
    hm=[]#2D heatmap
    hm_to_hm_en=[]#2D heatmap for drawing
    hm_en=[]#1D sorted heatmap
    colors=[]#array of colors to heatmap
    last_color=[255,0,0,128]#colors for heat map's legends 
    middle_color=[0,255,0,128]
    first_color=[0,0,255,128]
    def __init__(self,s,W,H,src_name,path_to_data,name):
        self.s=s
        self.W=W
        self.H=H
        self.src_name=src_name
        self.name=name
        self.path=path_to_data
        self.hm=np.zeros((W,H))
        
    def count_hm_en(self,hm_arr,W,H):
        hm_en=[]
        colors=[]
        for i in range(W):
            for j in range(H):
                if hm_arr[i][j]>0:
                    hm_en.append([hm_arr[i][j],i,j])
                    colors.append([0,0,0,128])
        hm_en.sort(key=lambda x: x[0])
        opt=len(hm_en)
        for j in range(3):
            colors[0][j]=self.first_color[j]
            colors[opt-1][j]=self.last_color[j]
        if((opt%2)==0):
            opt=int(opt/2)
            for j in range(3):
                colors[opt-1][j]=self.middle_color[j]
                colors[opt][j]=self.middle_color[j]
        else:
            opt=int(opt/2)
            for j in range(3):
                colors[opt][j]=self.middle_color[j]
        opt=opt-1 
        step_to_middle=[0,0,0,128]
        step_to_last=[0,0,0,128]
        if(opt>0):
            for i in range(3):
                step_to_middle[i]=(self.middle_color[i]-self.first_color[i])/opt
            for i in range(3):
                step_to_last[i]=(self.last_color[i]-self.middle_color[i])/opt
        for i in range(opt):
            for j in range(3):
                colors[i+1][j]=int(colors[0][j]+(step_to_middle[j])*(i+1))
        for i in range(opt):
            for j in range(3):
                colors[i+(opt+1)+1][j]=int(colors[opt+1][j]+(step_to_last[j])*(i+1))
        return hm_en,colors
    
    def draw_hm(self,hm_en,colors,s,name):
        im = Image.open(self.src_name)
        im == im.convert('RGBA')
        draw = ImageDraw.Draw(im,'RGBA')
        for i in range(len(colors)):
            x=hm_en[i][1]*s
            y=hm_en[i][2]*s
            draw.rectangle(((x,y),((x+s),(y+s))),fill=tuple(colors[i]))
        del draw
        im.save(self.path+name)
    
    def drawlegend(self):
            iw=100
            ih=400
            size=int(ih/len(self.colors))
            ih=len(self.colors)*size
            img = Image.new('RGB', (2*iw, ih),color=(255,255,255))
            draw=ImageDraw.Draw(img)
            draw.line([(0,0),(2*iw,0)],fill=(0,0,0),width=5)
            draw.line([(0,ih),(2*iw,ih)],fill=(0,0,0),width=5)
            fontsize=30
            font = ImageFont.truetype("arial.ttf", fontsize)
            draw.text((2*iw-70,ih-40),"min",font=font,fill=(0,0,0))
            draw.text((2*iw-70,5),"max",font=font,fill=(0,0,0))
            for i in range(len(self.colors)):
                color=[0,0,0]
                for j in range(3):
                    color[j]=self.colors[len(self.colors)-1-i][j]
                draw.rectangle(((0,i*size),(iw,(i+1)*size)),fill=tuple(color),width=1)
            img.save(self.path+'legend.png')
    
    def count_and_draw(self):
        self.hm_en,self.colors=self.count_hm_en(self.hm_to_hm_en,self.W,self.H)
        self.draw_hm(self.hm_en,self.colors,self.s,self.name)
class heatmap_with_text(heatmap):
    def draw_hm(self,hm_en,colors,s,name):
        super().draw_hm(hm_en,colors,s,name)
        im = Image.open(self.path+name)
        im == im.convert('RGBA')
        draw = ImageDraw.Draw(im,'RGBA')
        for i in range(len(colors)):
            x=hm_en[i][1]*s
            y=hm_en[i][2]*s
            W, H = (s,s)
            msg = str(hm_en[i][0])
            w, h = draw.textsize(msg)
            draw.text((x+(W-w)/2,y+(H-h)/2), msg, fill="black")
        del draw
        im.save(self.path+name)
class heatmap_with_hotest(heatmap):
    def draw_hotest(self,hm_en,colors,s,name):
        im = Image.open(self.src_name)
        im == im.convert('RGBA')
        draw = ImageDraw.Draw(im,'RGBA')
        slize=0.3
        sm=int(len(colors)*0.3)
        l=len(colors)-sm
        for i in range(l):
            x=hm_en[i+sm][1]*s
            y=hm_en[i+sm][2]*s
            draw.rectangle(((x,y),((x+s),(y+s))),fill=tuple(colors[i+sm]))
        del draw
        im.save(self.path+"hotest_"+name)
    
    def count_and_draw(self):
        super().count_and_draw()
        self.draw_hotest(self.hm_en,self.colors,self.s,self.name)
class heatmap_with_hotest_and_text(heatmap_with_hotest):
    def draw_hm(self,hm_en,colors,s,name):
        super().draw_hm(hm_en,colors,s,name)
        im = Image.open(self.path+name)
        im == im.convert('RGBA')
        draw = ImageDraw.Draw(im,'RGBA')
        for i in range(len(colors)):
            x=hm_en[i][1]*s
            y=hm_en[i][2]*s
            W, H = (s,s)
            msg = str(hm_en[i][0])
            w, h = draw.textsize(msg)
            draw.text((x+(W-w)/2,y+(H-h)/2), msg, fill="black")
        del draw
        im.save(self.path+name)
        
    def draw_hotest(self,hm_en,colors,s,name):
        im = Image.open(self.src_name)
        im == im.convert('RGBA')
        draw = ImageDraw.Draw(im,'RGBA')
        slize=0.3
        sm=int(len(colors)*0.3)
        l=len(colors)-sm
        for i in range(l):
            x=hm_en[i+sm][1]*s
            y=hm_en[i+sm][2]*s
            draw.rectangle(((x,y),((x+s),(y+s))),fill=tuple(colors[i+sm]))
        del draw
        im.save(self.path+"hotest_"+name)
    
    def count_and_draw(self):
        super().count_and_draw()
        self.draw_hotest(self.hm_en,self.colors,self.s,self.name)
class camstats(way,heatmap_with_hotest_and_text):#stats of one cam
    size=0#num of detected objects
    ways=[]#ways of objects
    ways_clear=[]#clear ways of objects
    W=0#width of image
    H=0#height of image
    detections=[]#places of detections
    hm_det=[]#heat map of detections
    hm_time=[]#data about time of objects in areas and their ammount
    hm_lengths=[]#data about curent lengths of objects in areas
    hm_lengths_current=[]#data about curent lengths of objects in areas
    hm_sizes={}#data about sizes of specific type of objects in areas and their amount
    types=[]#unique types of objects
    clusters_centres_lengths=[]#clusters centres of legths
    test_mode=True#test mode flag
    
    data_hist_hor=[[],[]]
    data_hist_vert=[[],[]]
    
    def __init__(self,path_to_data):
        self.path_to_data=path_to_data
        self.src_name=os.path.join(self.path_to_data,"frame.jpg")
        self.path_to_stats=os.path.join(self.path_to_data,"ways.txt")
        self.path_to_results=path_to_data+"/"
        im = Image.open(self.src_name)
        self.W, self.H = im.size
        self.load_ways()
        self.count_and_draw_detections()
        self.draw_lines(self.ways,"frame_with_ways_lines.png")
        self.count_and_draw_hm_det()
        self.count_and_draw_hm_time()
        self.count_and_draw_hm_lengths()
        #self.count_and_draw_hm_lengths_current()
        #self.count_and_draw_hms_size()
        self.clear()
        #self.count_and_draw_hist_hor()
        #self.count_and_draw_hist_vert()
    
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
    
    def load_ways_clear(self):
        self.ways_clear=[]
        with open(self.path_to_stats, 'r') as f:
            w = json.load(f)
            self.size=len(w)
            k=0
            for i in w:
                newway=way(np.transpose(np.array(i[1:])),i[0])
                newway.id=str(k)
                k+=1
                self.ways_clear.append(newway)
            if(self.test_mode==True):
                print("load_ways_clear - done")
    
    def draw_lines(self,ways,name):
        lines=[]
        im = Image.open(self.src_name)
        draw=ImageDraw.Draw(im)
        for i in ways:
            lines.append(i.centres)
        for line in lines:
            for i in range(len(line)-1):
                    draw.line([(line[i][0],line[i][1]),(line[i+1][0],line[i+1][1])],fill=(0,255,0),width=1)
        del draw
        im.save(self.path_to_results+name)
    
    def count_detections(self):
        self.detections=np.zeros([self.W, self.H],int)
        for way in self.ways:
            for coord in way.centres:
                self.detections[coord[0]][coord[1]]+=1
        
    def draw_detections(self):
        name="detections.png"
        detections_color=(255,255,255)
        im = Image.open(self.src_name)
        for i in range(self.W):
            for j in range(self.H):
                if self.detections[i][j]>0:
                    im.putpixel((i,j),detections_color)
        im.save(self.path_to_results+name)
    
    def count_and_draw_detections(self):
        self.count_detections()
        self.draw_detections()
        if(self.test_mode==True):
                print("count_and_draw_detections - done")
        
    
    def count_hm_det(self,s_det,W_det,H_det):
        hm_det=np.zeros([W_det,H_det],int)
        for i in range(W_det):
            for j in range(H_det):
                i_s=i*s_det
                j_s=j*s_det
                for i1 in range(s_det):
                    for j1 in range(s_det):
                        hm_det[i][j]=hm_det[i][j]+self.detections[i_s+i1][j_s+j1]
        return hm_det,copy.deepcopy(hm_det)
                        
    def count_and_draw_hm_det(self,s_modifier=4):
        s_det=int(math.gcd(self.W,self.H)/s_modifier)
        W_det=int(self.W/s_det)
        H_det=int(self.H/s_det)
        self.hm_det=heatmap_with_hotest(s_det,W_det,H_det,self.src_name,self.path_to_results,"heatmap_detections.png")
        self.hm_det.hm,self.hm_det.hm_to_hm_en=self.count_hm_det(s_det,W_det,H_det)
        self.hm_det.count_and_draw()
        if(self.test_mode==True):
                print("count_and_draw_hm_det - done")
        
    def count_hm_time(self,s_time,W_time,H_time):
        hm_time=[]
        for i in range(W_time):
            hm_time.append([])
            for j in range(H_time):
                hm_time[i].append([0,0])
        for i in self.ways:
            start=0
            while(start<(len(i.centres)-1)):
                end=start
                X_cell=int(i.centres[start][0]/s_time)
                Y_cell=int(i.centres[start][1]/s_time)
                while((X_cell==int(i.centres[end][0]/s_time))and(Y_cell==int(i.centres[end][1]/s_time))):
                    end=end+1
                    if(end==(len(i.centres)-1)):
                        break
                fps=30
                T=(i.frames[end-1]-i.frames[start])/fps
                if(T!=0):
                    hm_time[X_cell][Y_cell][0]=hm_time[X_cell][Y_cell][0]+1
                    hm_time[X_cell][Y_cell][1]=hm_time[X_cell][Y_cell][1]+T
                start=end
        hm_time_avg=[]
        for i in range(W_time):
            hm_time_avg.append([])
            for j in range(H_time):
                if hm_time[i][j][0]>0:
                    hm_time_avg[i].append(round((hm_time[i][j][1]/hm_time[i][j][0]),2))
                else: 
                    hm_time_avg[i].append(0)
        return hm_time,hm_time_avg
    
    def count_and_draw_hm_time(self,s_modifier=2):
        s_time=int(math.gcd(self.W,self.H)/s_modifier)
        W_time=int(self.W/s_time)
        H_time=int(self.H/s_time)
        self.hm_time=heatmap_with_text(s_time,W_time,H_time,self.src_name,self.path_to_results,"heatmap_avg_time.png")
        self.hm_time.hm,self.hm_time.hm_to_hm_en=self.count_hm_time(s_time,W_time,H_time)
        self.hm_time.count_and_draw()
        if(self.test_mode==True):
                print("count_and_draw_hm_time - done")
    
    def count_hm_lengths(self,s_lengths,W_lengths,H_lengths):
        hm_lengths=[]
        for i in range(W_lengths):
            hm_lengths.append([])
            for j in range(H_lengths):
                hm_lengths[i].append([0,0])
        for i in self.ways:
            if i.size>0:
                for j in range(i.size):
                    X_cell=int(i.centres[j][0]/s_lengths)
                    Y_cell=int(i.centres[j][1]/s_lengths)
                    hm_lengths[X_cell][Y_cell][0]=hm_lengths[X_cell][Y_cell][0]+1
                    hm_lengths[X_cell][Y_cell][1]=hm_lengths[X_cell][Y_cell][1]+i.length
        hm_lengths_avg=[]
        for i in range(W_lengths):
            hm_lengths_avg.append([])
            for j in range(H_lengths):
                if hm_lengths[i][j][0]>0:
                    hm_lengths_avg[i].append(int(hm_lengths[i][j][1]/hm_lengths[i][j][0]))
                else: 
                    hm_lengths_avg[i].append(0)
        return copy.deepcopy(hm_lengths),copy.deepcopy(hm_lengths_avg)
    
    def count_and_draw_hm_lengths(self,s_modifier=2):
        s_lengths=int(math.gcd(self.W,self.H)/s_modifier)
        W_lengths=int(self.W/s_lengths)
        H_lengths=int(self.H/s_lengths)
        self.hm_lengths=heatmap_with_hotest_and_text(s_lengths,W_lengths,H_lengths,self.src_name,self.path_to_results,"heatmap_lengths.png")
        self.hm_lengths.hm,self.hm_lengths.hm_to_hm_en=self.count_hm_lengths(s_lengths,W_lengths,H_lengths)
        self.hm_lengths.count_and_draw()
        if(self.test_mode==True):
                print("count_and_draw_hm_lengths - done")
    
    def count_hm_lengths_current(self,s_lengths,W_lengths,H_lengths):
        hm_lengths=[]
        for i in range(W_lengths):
            hm_lengths.append([])
            for j in range(H_lengths):
                hm_lengths[i].append([0,0])
        for i in self.ways:
            if i.size>0:
                for j in range(i.size):
                    X_cell=int(i.centres[j][0]/s_lengths)
                    Y_cell=int(i.centres[j][1]/s_lengths)
                    hm_lengths[X_cell][Y_cell][0]=hm_lengths[X_cell][Y_cell][0]+1
                    hm_lengths[X_cell][Y_cell][1]=hm_lengths[X_cell][Y_cell][1]+i.lengths[j]
        hm_lengths_avg=[]
        for i in range(W_lengths):
            hm_lengths_avg.append([])
            for j in range(H_lengths):
                if hm_lengths[i][j][0]>0:
                    hm_lengths_avg[i].append(int(hm_lengths[i][j][1]/hm_lengths[i][j][0]))
                else: 
                    hm_lengths_avg[i].append(0)
        return copy.deepcopy(hm_lengths),copy.deepcopy(hm_lengths_avg)
    
    def count_and_draw_hm_lengths_current(self,s_modifier=2):
        s_lengths=int(math.gcd(self.W,self.H)/s_modifier)
        W_lengths=int(self.W/s_lengths)
        H_lengths=int(self.H/s_lengths)
        self.hm_lengths_current=heatmap_with_hotest_and_text(s_lengths,W_lengths,H_lengths,self.src_name,self.path_to_results,"heatmap_length_current.png")
        self.hm_lengths_current.hm,self.hm_lengths_current.hm_to_hm_en=self.count_hm_lengths_current(s_lengths,W_lengths,H_lengths)
        self.hm_lengths_current.count_and_draw()
        if(self.test_mode==True):
                print("count_and_draw_hm_lengths_current - done")
    
    def count_hm_size(self,s_size,W_size,H_size,o_t=2):
        hm_size=[]
        for i in range(W_size):
            hm_size.append([])
            for j in range(H_size):
                hm_size[i].append([0,0])
        for i in self.ways:
            for j in range(i.size):
                if (i.object_type==o_t):
                    X_cell=int(i.centres[j][0]/s_size)
                    Y_cell=int(i.centres[j][1]/s_size)
                    hm_size[X_cell][Y_cell][0]=hm_size[X_cell][Y_cell][0]+1
                    hm_size[X_cell][Y_cell][1]=hm_size[X_cell][Y_cell][1]+i.sizes[j][0]*i.sizes[j][1]
                else:
                    break
        hm_size_avg=[]
        for i in range(W_size):
            hm_size_avg.append([])
            for j in range(H_size):
                if hm_size[i][j][0]>0:
                    hm_size_avg[i].append(int(hm_size[i][j][1]/hm_size[i][j][0]))
                else: 
                    hm_size_avg[i].append(0)
        return copy.deepcopy(hm_size),copy.deepcopy(hm_size_avg)
    
    def count_and_draw_hms_size(self,s_modifier=2):
        s_size=int(math.gcd(self.W,self.H)/s_modifier)
        W_size=int(self.W/s_size)
        H_size=int(self.H/s_size)
        for i in self.types:
            hm_size=heatmap_with_text(s_size,W_size,H_size,self.src_name,self.path_to_results,"sizes/heatmap_avg_size_of_"+names[i].rstrip()+"s.png")
            hm_size.hm,hm_size.hm_to_hm_en=self.count_hm_size(s_size,W_size,H_size,i)
            hm_size.count_and_draw()
            self.hm_sizes[i]=hm_size
        if(self.test_mode==True):
                print("count_and_draw_hms_size - done")

    def remove_ways_empty(self):
        sz=len(self.ways_clear)
        j=0
        while (j<sz):
            if self.ways_clear[j].size==0:
                self.ways_clear.pop(j)
                sz=sz-1
            else:
                j=j+1
    
    def clear_from_cold(self):
        slize=0.3
        num=int(len(self.hm_det.hm_en)*slize)
        etalon=self.hm_det.hm_en[num][0]
        s=self.hm_det.s
        for way in self.ways_clear:
            sz=way.size
            j=0
            while (j<sz):
                X_cell=int(way.centres[j][0]/s)
                Y_cell=int(way.centres[j][1]/s)
                if(self.hm_det.hm[X_cell][Y_cell]<etalon):
                    way.delete_element(j)
                    sz=way.size
                else:
                    j=j+1
        self.remove_ways_empty()
        if(self.test_mode==True):
            print("clear_from_cold - done")
                
        
    def clear_from_anomaly_sizes(self):
        top_slize=2
        low_slize=0.5
        for way in self.ways_clear:
            sz=way.size
            o_t=way.object_type
            s=self.hm_sizes[o_t].s
            j=0
            while (j<sz):
                X_cell=int(way.centres[j][0]/s)
                Y_cell=int(way.centres[j][1]/s)
                etalon=self.hm_sizes[o_t].hm_to_hm_en[X_cell][Y_cell]
                S=way.sizes[j][0]*way.sizes[j][1]
                if((S>(top_slize*etalon))or(S<(low_slize*etalon))):
                    way.delete_element(j)
                    sz=way.size
                else:
                    j=j+1
        self.remove_ways_empty()
        if(self.test_mode==True):
            print("clear_from_anomaly_sizes - done")
        
        
    def clear_from_short(self):
        nums=[]
        for way in self.ways_clear:
            if(way.length>0):
                nums.append(way.length)
        nums=np.array(nums).reshape(-1, 1)
        ms = MeanShift().fit(nums)
        cluster_centers = ms.cluster_centers_
        self.clusters_centres_lengths=copy.copy(cluster_centers)
        etalon=0
        for i in cluster_centers:
            etalon+=i
        etalon=int(etalon/len(cluster_centers))
        sz=len(self.ways_clear)
        j=0
        while (j<sz):
            if self.ways_clear[j].length<etalon:
                self.ways_clear.pop(j)
                sz=sz-1
            else:
                j=j+1
        if(self.test_mode==True):
            print("clear_from_short - done")
        
    
    def func(self,t,y0,v0,a):
        return y0 + v0*t + 0.5*a*t**2
    
    def count_lines_smooth(self):
        lines_clear_smooth=[]
        for i in self.ways_clear:
            lines_clear_smooth.append(i.centres)
        for i in lines_clear_smooth:
            if(len(i)>2):
                i=np.transpose(i)
                popt, pcov = curve_fit(self.func, i[0], i[1], p0=[self.W,self.H,self.H], absolute_sigma=True)
                #plt.plot(i[0], self.func(i[0], *popt))
                i[1]=self.func(i[0], *popt)
                for j in i[1]:
                    j=int(j)
       
    def count_and_draw_lines_clear_smooth(self):
        self.count_lines_smooth()
        self.draw_lines(self.ways_clear,"frame_with_ways_lines_clear_smooth.png")
        if(self.test_mode==True):
                print("count_and_draw_lines_smooth - done")
    
    def save(self):
        save=[]
        for i in self.ways_clear:
            save.append(i.to_save())
        with open(self.path_to_results+"ways_clear.txt", 'w') as f:
            json.dump(save, f)
    
    def clear(self):
        self.load_ways_clear()
        #self.clear_from_cold()
        #self.clear_from_anomaly_sizes()
        self.clear_from_short()
        self.draw_lines(self.ways_clear,"frame_with_ways_lines_clear.png")
        self.count_and_draw_lines_clear_smooth()
        self.save()
        if(self.test_mode==True):
            print("clear - done")
    
    def __str__(self):
        rep = "Size="+str(self.size)+"\n"
        for way in self.ways:
            rep+=str(way)
        return rep
    
    def count_and_draw_hist_hor(self):
        #get_data
        #######################
        s=math.gcd(self.W,self.H)
        data_for_hist_hor=[]
        for i in range(int(self.W/s)):
            data_for_hist_hor.append([0,0])
        for i in self.ways:
            if i.size>0:
                for j in range(i.size-1):
                    cell=int(i.centres[j][0]/s)
                    data_for_hist_hor[cell][0]+=i.lengths[j]
                    data_for_hist_hor[cell][1]+=1
        self.data_hist_hor=[[],[]]
        for i in range(int(self.W/s)):
            self.data_hist_hor[0].append(data_for_hist_hor[i][0]/data_for_hist_hor[i][1])
            self.data_hist_hor[1].append(int(i*s+s/2))
        #draw hist
        #################################
        plt.bar(self.data_hist_hor[1],self.data_hist_hor[0], width=s,edgecolor="black")
        plt.show()
        
    def count_and_draw_hist_vert(self):
        #get_data
        #######################
        s=math.gcd(self.W,self.H)
        data_for_hist_vert=[]
        for i in range(int(self.H/s)):
            data_for_hist_vert.append([0,0])
        for i in self.ways:
            if i.size>0:
                for j in range(i.size-1):
                    cell=int(i.centres[j][1]/s)
                    data_for_hist_vert[cell][0]+=i.lengths[j]
                    data_for_hist_vert[cell][1]+=1
        self.data_hist_vert=[[],[]]
        for i in range(int(self.H/s)):
            self.data_hist_vert[0].append(data_for_hist_vert[i][0]/data_for_hist_vert[i][1])
            self.data_hist_vert[1].append(int(i*s+s/2))
        #draw hist
        #################################
        plt.barh(self.data_hist_vert[1],self.data_hist_vert[0],height=s,edgecolor="black")
        plt.show()