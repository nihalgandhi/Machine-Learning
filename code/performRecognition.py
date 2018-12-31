# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image 
im = cv2.imread("test2.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
yo, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

(ctrs, rects) = zip(*sorted(zip(ctrs, rects), key=lambda b: b[1][0], reverse=False))

num = []
#i = 0;
# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    #print(leng,pt1,pt2,im_th.shape)
    a,b = im_th.shape
    p1s = min(pt1%a,(pt1+leng)%a)
    p1e =  max(pt1%a,(pt1+leng)%a)
    p2s = min(pt2%b, (pt2 + leng)%b)
    p2e = max(pt2%b, (pt2 + leng)%b)
    roi = im_th[p1s:p1e, p2s:p2e]
    # roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
    # Resize the image
    #print(roi.shape)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA,)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    #nbr[0] += i
    #i += 1
    #num.append(nbr[0])
    if nbr == 14:
        nbr=nbr-13
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    elif nbr == 10:
        cv2.putText(im, str('+'), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    elif nbr == 11:
        cv2.putText(im, str('-'), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    elif nbr == 12:
        cv2.putText(im, str('x'), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    elif nbr == 13:
        cv2.putText(im, str('/'), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    else:
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    num.append(nbr[0])
####################################################################################
c=0
d=0
arr=[]
for i in range(len(num)):
    if num[i]<10:
        c=c*10+num[i]
    else:
        arr.append(c)
        c=0
        arr.append(-num[i])
arr.append(c)
print(arr)
#division
new = []
j=0
i=0
while i<len(arr):
    if arr[i] == -13:
        new[j-1] = arr[i-1]/arr[i+1]
        i = i+2
        j += 1
    else:
        new.append(arr[i])
        j += 1
        i += 1
#print(new)
#multiply
arr = []
j=0
i=0
while i<len(new):
    if new[i] == -12:
        arr[j-1] = new[i-1]*new[i+1]
        i = i+2
        j += 1
    else:
        arr.append(new[i])
        j += 1
        i += 1
#print(arr)
#addition
new = []
j=0
i=0
while i<len(arr):
    if arr[i] == -10:
        new[j-1] = arr[i-1]+arr[i+1]
        i = i+2
        j += 1
    else:
        new.append(arr[i])
        j += 1
        i += 1
#print(new)
#subtraction
arr = []
j=0
i=0
while i<len(new):
    if new[i] == -11:
        arr[j-1] = new[i-1]-new[i+1]
        i = i+2
        j += 1
    else:
        arr.append(new[i])
        j += 1
        i += 1

print(arr)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()

