# Import the modules
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter
import cv2

# Extract the features and labels
features = np.load('digit_features.npy')
labels = np.load('digit_labels.npy')
labels_list = labels.tolist()

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
    print("ho rha hai!!")

print("Count of digits in dataset", Counter(labels))

##############################################################################################
# Read the input image
im = cv2.imread("plusnew.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
yo, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for i in range(300):
    for rect in rects:
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        a, b = im_th.shape
        p1s = min(pt1 % a, (pt1 + leng) % a)
        p1e = max(pt1 % a, (pt1 + leng) % a)
        p2s = min(pt2 % b, (pt2 + leng) % b)
        p2e = max(pt2 % b, (pt2 + leng) % b)
        roi = im_th[p1s:p1e, p2s:p2e]
        #roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(roi_hog_fd)
        labels_list.append(10)
    print("first part")

print("Count of digits in dataset", Counter(labels))
##################################################################################################

##############################################################################################
# Read the input image
im = cv2.imread("add2.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
yo, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for i in range(400):
    for rect in rects:
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        a, b = im_th.shape
        p1s = min(pt1 % a, (pt1 + leng) % a)
        p1e = max(pt1 % a, (pt1 + leng) % a)
        p2s = min(pt2 % b, (pt2 + leng) % b)
        p2e = max(pt2 % b, (pt2 + leng) % b)
        roi = im_th[p1s:p1e, p2s:p2e]
        #roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(roi_hog_fd)
        labels_list.append(10)
    print("first part")

print("Count of digits in dataset", Counter(labels))
##################################################################################################

##############################################################################################
# Read the input image
im = cv2.imread("sub1.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
yo, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for i in range(300):
    for rect in rects:
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(roi_hog_fd)
        labels_list.append(11)
    print("second part")


print("Count of digits in dataset", Counter(labels))
##################################################################################################

##############################################################################################
# Read the input image
im = cv2.imread("mul.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
yo, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for i in range(900):
    for rect in rects:
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        a, b = im_th.shape
        p1s = min(pt1 % a, (pt1 + leng) % a)
        p1e = max(pt1 % a, (pt1 + leng) % a)
        p2s = min(pt2 % b, (pt2 + leng) % b)
        p2e = max(pt2 % b, (pt2 + leng) % b)
        roi = im_th[p1s:p1e, p2s:p2e]
        #roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(roi_hog_fd)
        labels_list.append(12)
    print("third part")


print("Count of digits in dataset", Counter(labels))
##################################################################################################

##############################################################################################
# Read the input image
im = cv2.imread("div.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
yo, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for i in range(720):
    for rect in rects:
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        a, b = im_th.shape
        p1s = min(pt1 % a, (pt1 + leng) % a)
        p1e = max(pt1 % a, (pt1 + leng) % a)
        p2s = min(pt2 % b, (pt2 + leng) % b)
        p2e = max(pt2 % b, (pt2 + leng) % b)
        roi = im_th[p1s:p1e, p2s:p2e]
        #roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(roi_hog_fd)
        labels_list.append(13)
    print("fourth part")


print("Count of digits in dataset", Counter(labels_list))
##################################################################################################

##############################################################################################
# Read the input image
im = cv2.imread("one1.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
yo, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for i in range(300):
    for rect in rects:
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        a, b = im_th.shape
        p1s = min(pt1 % a, (pt1 + leng) % a)
        p1e = max(pt1 % a, (pt1 + leng) % a)
        p2s = min(pt2 % b, (pt2 + leng) % b)
        p2e = max(pt2 % b, (pt2 + leng) % b)
        roi = im_th[p1s:p1e, p2s:p2e]
        #roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(roi_hog_fd)
        labels_list.append(14)
    print("fifth part")


print("Count of digits in dataset", Counter(labels_list))
##################################################################################################

# Create an linear SVM object
clf = LinearSVC()

labels = np.array(labels_list)
hog_features = np.array(list_hog_fd, 'float64')
# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)