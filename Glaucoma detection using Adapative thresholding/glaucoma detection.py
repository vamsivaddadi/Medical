from scipy import signal
import cv2 as cv
from pylab import*
from scipy import signal
import numpy as np


clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))

image = cv.imread('D:/Work/Retinopathy/eye dataset/Drishti-GS1_files/Drishti-GS1_files/Drishti-GS1_files/Training/Images/drishtiGS_022.png')
#image = cv.imread('D:/Work/Retinopathy/eye dataset/Acimr Glaucoma/Database/Database/Images/Im324_g_ACRIMA.jpg')

###drishti
roi = image[400:1400, 500:1600, :]

###drions roi
#roi = image[70:350, 150:400, :]

_, g, _ = cv.split(image)


g = clahe.apply(g)
m = 60
filter = signal.gaussian(m,std = 6)
filter = filter/sum(filter)
stdf = filter.std()

######### Disc

_, _, r = cv.split(roi)

red_pp = r - r.mean() - r.std()
threshold = 0.5*m - 2*stdf - red_pp.std()

_,thresh = cv.threshold(red_pp, threshold, 255, cv.THRESH_BINARY)
thresh = thresh.astype(np.uint8)
#print(thresh.shape)
ih,iw = thresh.shape

disc_c1 = cv.morphologyEx(thresh, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2)), iterations = 1)
disc_o1 = cv.morphologyEx(disc_c1, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)), iterations = 1)
disc_c2 = cv.morphologyEx(disc_o1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,21)), iterations = 1)
disc_o2 = cv.morphologyEx(disc_c2, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(21,1)), iterations = 1)
disc_c3 = cv.morphologyEx(disc_o2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(33,33)), iterations = 1)
disc_o3 = cv.morphologyEx(disc_c3, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(43,43)), iterations = 1)            

disc_edges = cv.Canny(disc_o3,70, 35)

contours,_ = cv.findContours(disc_edges,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

largest = max(contours, key=cv.contourArea)
con = np.zeros((ih,iw), dtype=np.uint8)
disc_con = cv.drawContours(con,largest, -1, (255,255,255), 1)


disc_el = cv.fitEllipse(largest)
(xc,yc),(d1,d2),angle = disc_el

disc = cv.ellipse(roi,disc_el,(255,255,255),1)

d_dia = int(max(d1,d2))
print('diameter of the disc',d_dia)


####### Cup
green_pp = g - g.mean() - g.std()
mg = green_pp.mean()
sdg = green_pp.std()
green_threshold = 0.5*m + 2*stdf + 2*sdg + mg

histg, bins = np.histogram(green_pp.ravel(), 256, [0, 256])

green_hist_smooth = np.convolve(filter,histg)

cup_ret, cup_thresh = cv.threshold(green_pp, green_threshold-3, 255, cv.THRESH_BINARY)

cup_thresh = cup_thresh.astype(np.uint8)

cup_c1 = cv.morphologyEx(cup_thresh, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2)), iterations = 1)
cup_o1 = cv.morphologyEx(cup_c1, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)), iterations = 1)
cup_c2 = cv.morphologyEx(cup_o1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,15)), iterations = 1)
cup_o2 = cv.morphologyEx(cup_c2, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(17,1)), iterations = 1)
cup_c3 = cv.morphologyEx(cup_o2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(30,30)), iterations = 1)	
cup_o3 = cv.morphologyEx(cup_c3, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(35,35)), iterations = 1)


edges = cv.Canny(cup_o3, 70, 35)
edges = edges[400:1400, 500:1600]
cup_contour, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#c_im = cv.drawContours(disc, cup_contour, -1, (255,255,255), 1)
#cc = np.zeros((ih,iw), dtype=np.uint8)

c_max = max(cup_contour, key=cv.contourArea)
c_el = cv.fitEllipse(c_max)

cup = cv.ellipse(disc,c_el, (255,123,0),1)

(xc, yc),(d1, d2), angle = c_el 
c_dia = int(max(d1,d2))
print('diameter of the cup',c_dia)


cdr = c_dia/d_dia
print(cdr) 

if cdr >= 0.8:
    diagnosis = 'Severe'
    print('Risk of Glaucoma is ',diagnosis)

elif cdr >=0.5 and cdr <0.8:
                    
    diagnosis = 'Moderate'
    print('Risk of Glaucoma is ',diagnosis)
    
                   
elif cdr >0.3 and cdr <0.5:
    diagnosis = 'Mild'
    print('Risk of Glaucoma is ',diagnosis)

else:
    diagnosis = 'None'
    print('Risk of Glaucoma is ',diagnosis)




#cv.imshow('cup thresh',cup_thresh)
#cv.imshow('cup and disc',cup)
cv.waitKey(0)
cv.destroyAllWindows    