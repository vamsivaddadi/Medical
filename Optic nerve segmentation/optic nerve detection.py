import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numpy.lib import tile
from skimage.transform import radon
#import imutils
from PIL import Image
from skimage.filters.thresholding import try_all_threshold, threshold_isodata
from skimage.morphology import area_opening

def resize(image):
    width = round(image.shape[1]*1/3)
    height = round(image.shape[0]*1/3)
    image = cv.resize(image,(height,width))
    return image


#dr = cv.imread('1.png')
dr = cv.imread('D:/Work/Retinopathy/eye dataset/Diabetic Retinopathy preprocessing/diabetic_retinopathy/15_dr.jpg')
#dr = cv.imread('D:/work/retinopathy/eye dataset/acimr glaucoma/Database/Database/Images/Im061_ACRIMA.jpg')
#dr = cv.imread('D:/Work/Retinopathy/eye dataset/Drishti-GS1_files/Drishti-GS1_files/Drishti-GS1_files/Training/Images/drishtiGS_033.png')
#dr = cv.imread('Sample data/Im002_ACRIMA.jpg')
#dr = cv.imread('D:/Work/Retinopathy/eye dataset/Drishti-GS1_files/Drishti-GS1_files/Drishti-GS1_files/Training/Images/drishtiGS_002.png')
#dr = cv.imread('D:/Work/Retinopathy/eye dataset/Diabetic Retinopathy preprocessing/diabetic_retinopathy/03_dr.jpg')


dr = resize(dr)


gray = cv.cvtColor(dr, cv.COLOR_BGR2GRAY)

green = dr[:,:,1]
inverted_green = 255 - green

hist , bins = np.histogram(dr.flatten(),128,[0,256])
cdf = hist.cumsum()
cdf_norm = cdf * float(hist.max() / cdf.max())

hist1, bins1 = np.histogram(dr.flatten(),128,[0,128])
cdf1 = hist1.cumsum()
cdf_norm1 = cdf1 * float(hist1.max() / cdf1.max())

equ_gray = cv.equalizeHist(gray)
equ_green = cv.equalizeHist(green)
"""plt.plot(cdf_norm, color = 'r')
plt.subplot(2,1,1)
plt.hist(dr.flatten(),128,[0,256],color='b')

plt.subplot(2,1,2)
plt.plot(cdf_norm1, color = 'r')
plt.hist(dr.flatten(),128,[0,256],color='b')
#plt.show()"""

clahe = cv.createCLAHE(clipLimit=6, tileGridSize=(9,9))
clahe_gray = clahe.apply(gray)
clahe_green = clahe.apply(green)


#cv.imwrite('clahe gray.png',clahe_gray)
#cv.imwrite('clahe green.png',clahe_green)



#filtering
def filter(image,name):
    
    #gauss = cv.GaussianBlur(image, (9,9),0)
    average = cv.blur(image,(9,9))
    #cv.imshow(name + 'gauss',average)
    
    return average


#contrast_stretch(clahe_gray)

average_gray = filter(clahe_gray,'gray')
average_green = filter(clahe_green,'Green')
#print('gray ave shape',average_gray.shape)
#print('clahe gray shape',clahe_gray.shape)



process_gray = cv.subtract(average_gray,clahe_gray)
process_green = cv.subtract(average_green,clahe_green)



#### all thresholds 
def threshold(image,name):
    fig, ax = try_all_threshold(image, figsize=(10, 8), verbose=False)
    plt.show()


#### ISODATA thershold
def iso_thresh(image, name):
    
    threshold = threshold_isodata(image)
    binary = image > threshold
    #print(threshold)
    #print(name + 'threshold value is', threshold)
    #plt.title(name)
    #plt.imshow(binary,cmap= plt.cm.Greys_r)
    #plt.show()
    #plt.imsave('threshold images/isodata ' + name + 'l3.png', binary, cmap = plt.cm.Greys_r)
    return threshold 


#threshold(process_gray,'gray process')
#threshold(process_green,'green processed')


gray_thresh = iso_thresh(process_gray, 'gray') -2
print('grey threshold is', gray_thresh)
green_thresh = iso_thresh(process_green,'green') - 2
print('green threshold',green_thresh)


#iso_data_gray = cv.imread('Threshold images/isodata grayl3.png')
#iso_data_green = cv.imread('Threshold images/isodata greenl3.png')
#cv.imshow('iso data gray', iso_data_gray)
#cv.imshow('iso data green',iso_data_green)



## binary threshold

ret1, bw_gray = cv.threshold(process_gray, gray_thresh, 255 , cv.THRESH_BINARY)
ret2, bw_green = cv.threshold(process_green, green_thresh, 255, cv.THRESH_BINARY)


#fig, (ax1, ax2) = plt.subplots(1,2 ,figsize = (10, 4))
#ax1.set_title('bw gray')
#ax1.imshow(bw_gray, cmap = plt.cm.Greys_r)
#ax2.set_title('bw green')
#ax2.imshow(bw_green, cmap = plt.cm.Greys_r)
#plt.show()


close_kernal = np.ones((3,3), np.uint8)


### Morph opening skimage 
###deleting pixcels less than 35
pixcel = 35
pix_gray = area_opening(bw_gray,pixcel)
pix_green = area_opening(bw_green,pixcel)

### Opening
open_kernal = np.ones((2,2), np.uint8)
bw_gray_open = cv.morphologyEx(pix_gray, cv.MORPH_OPEN, kernel = open_kernal)
bw_green_open = cv.morphologyEx(pix_green, cv.MORPH_OPEN, kernel = open_kernal)

### Closing
bw_gray_close = cv.morphologyEx(pix_gray, cv.MORPH_CLOSE, kernel= close_kernal)
bw_green_close = cv.morphologyEx(pix_green, cv.MORPH_CLOSE, kernel=close_kernal)

### Masking
def mask(image):
    ret , mask = cv.threshold(image, 12,255, cv.THRESH_BINARY_INV)
    #cv.imshow('Mask',mask)
    return mask

def mask_subtract(image1, image2, name):
    out = cv.subtract(image1,image2)
    return out #print('out of '+name)

def median(image):
    final = cv.medianBlur(image,5)
    return final


#eye_mask = mask(gray)
"""mask=np.zeros(dr.shape[:], dtype=np.uint8)
mask=cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
ret,eye_mask=cv.threshold(mask,12,255,cv.THRESH_BINARY)"""

green_mask = mask(green)


"""post_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
opened_mask = cv.morphologyEx(eyemask, cv.MORPH_OPEN, post_kernel)
masked_img = cv.bitwise_and(bw_gray, bw_gray, mask=opened_mask)"""

#gray_median = median(bw_gray)
#green_median = median(bw_green)

green_vessel = cv.subtract(bw_green_close,green_mask)
gray_vessel = cv.subtract(bw_gray_close,green_mask)
#cv.imwrite('sample data/gray vessels.jpg',gray_vessel)
#cv.imwrite('sample data/green vessels.jpg',green_vessel)

close_vesssel = np.ones((7,7), np.uint8)
green_vessel_close = cv.morphologyEx(green_vessel, cv.MORPH_CLOSE, kernel=close_kernal)

open_vessel = np.ones((9,9), np.uint8)
green_vessel_open = cv.morphologyEx(green_vessel, cv.MORPH_OPEN, kernel = open_kernal)


#inv_ret,inverse_vessel = cv.threshold(vessel, 255, 0, cv.THRESH_BINARY_INV)

inverse_green_vessel = np.invert(green_vessel_close) 

#####subtract vessels from threshold image
no_vessel_gray = cv.subtract(bw_gray,gray_vessel)
no_vessel_green = cv.subtract(bw_green, green_vessel)
#cv.imwrite('sample data/no vessel gray.jpg',no_vessel_gray)
#cv.imwrite('sample data/no vessel green.jpg',no_vessel_gray)


#median_gray = median(vessel)
median_green = median(green_vessel_close)
#cv.imwrite('sample data/median green.jpg',median_green)

####edges
#gray_edges = cv.Canny(median_gray, 70,35)
green_edges = cv.Canny(median_green, 70, 35)


#contours
contours, hierarchy = cv.findContours(green_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

copy = dr.copy()

vessels = cv.drawContours(copy,contours, -1, (255,255,255), 1)

#cv.imwrite('sample data/vessel contours.jpg',vessels)

b=np.multiply(dr[:,:,0],green_vessel)
g=np.multiply(dr[:,:,1],green_vessel)
r=np.multiply(dr[:,:,2],green_vessel)

# img_final combines these 3 seperated channels to get the final Fundus image
img_final=np.zeros(dr.shape[:])
img_final=np.uint8(img_final)
img_final[:,:,0]=b
img_final[:,:,1]=g
img_final[:,:,2]=r
img_final = cv.merge([b,g,r])
#close_kernal = np.ones((3,3), np.uint8)

image = cv.bitwise_and(copy, copy, mask=median_green)
#cv.imshow('image bitwise',image)

minus = cv.subtract(dr,image)
cv.imwrite('sample data/con.jpg',minus)
#cv.imshow('no vessels',minus)

#cv.imshow('filtered', fil_img)
#cv.imshow('original',dr)

#cv.imshow('gray', gray)
#cv.imshow('green', green)

#cv.imshow('threshold gray',bw_gray)
#cv.imshow('threshold green',bw_green)

#cv.imshow('clahe gray', clahe_gray)
#cv.imshow('clahe green', clahe_green)

#cv.imshow('average',average_gray)

#cv.imshow('gray pro',process_gray)
#cv.imshow('green pro',process_green)

#cv.imshow('bw gray', bw_gray)
#cv.imshow('bw green', bw_green)

#cv.imshow('bw gray opening',bw_gray_open)
#cv.imshow('bw green opening', bw_green_open)

#cv.imshow('bw gray close', bw_gray_close)
#cv.imshow('bw green close', bw_green_close)

#cv.imshow('pix gray',pix_gray)
#cv.imshow('pix green',pix_green)


#cv.imshow('vessels gray', gray_vessel)
#cv.imshow('vessel green',green_vessel)


#cv.imshow('mask',green_mask)

#cv.imshow('vessel close',vessel1_close)
#cv.imshow('vessel open',vessel1_open)


#cv.imshow('inverse green', inverse_vessel1)


#cv.imshow('no vessels gray', no_vessel_gray)
#cv.imshow('no vessels green', no_vessel_green)

#cv.imshow('median gray',median_gray)
#cv.imshow('median green',median_green)


#cv.imshow('canny gray',gray_edges)
#cv.imshow('canny green',green_edges)

#cv.imshow('final image',img_final)

cv.imshow('vessel contours',vessels)

cv.waitKey(0)
cv.destroyAllWindows()
