import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('ecg.jpg')
gray = 255 - cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# prepare a mask using Otsu threshold, then copy from original. this removes some noise
__, bw = cv2.threshold(cv2.dilate(gray, None), 128, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
gray = cv2.bitwise_and(gray, bw)

# make copy of the low-noise underlined image
grayu = gray.copy()
imcpy = im.copy()

# scan each row and remove lines
for row in range(gray.shape[0]):
    avg = np.average(gray[row, :] > 16)
    if avg > 0.9:
        cv2.line(im, (0, row), (gray.shape[1]-1, row), (0, 0, 255))
        cv2.line(gray, (0, row), (gray.shape[1]-1, row), (0, 0, 0), 1)

cont = gray.copy()
graycpy = gray.copy()


# after contour processing, the residual will contain small contours
residual = gray.copy()

# find contours
contours, hierarchy = cv2.findContours(cont, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    # find the boundingbox of the contour
    x, y, w, h = cv2.boundingRect(contours[i])
    if 40 < h:
        cv2.drawContours(im, contours, i, (0, 255, 0), -1)
        # if boundingbox height is higher than threshold, remove the contour from residual image
        cv2.drawContours(residual, contours, i, (0, 0, 0), -1)
    else:
        cv2.drawContours(im, contours, i, (255, 0, 0), -1)
        # if boundingbox height is less than or equal to threshold, remove the contour gray image
        cv2.drawContours(gray, contours, i, (0, 0, 0), -1)

# now the residual only contains small contours. open it to remove thin lines
st = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, st)
# prepare a mask for residual components
__, residual = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY)
# combine the residuals. we still need to link the residuals
combined = cv2.bitwise_or(cv2.bitwise_and(graycpy, residual), gray)
# link the residuals
st = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
linked = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, st)
# prepare a msak from linked image
__, mask = cv2.threshold(linked, 0, 255, cv2.THRESH_BINARY)
# copy region from low-noise underlined image
clean = 255 - cv2.bitwise_and(grayu, mask)
cv2.imwrite('image.jpg', clean)

# Repair image
repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
result = 255 - cv2.morphologyEx(255 - clean, cv2.MORPH_CLOSE, repair_kernel, iterations=2)
cv2.imwrite('image.jpg', result)

image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# cv2.imwrite('image.jpg', image)

# # Remove vertical
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
# detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
# cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     cv2.drawContours(image, [c], -1, (255,255,255), 2)
# cv2.imwrite('image.jpg', image)

# # repair image
# kernel_line = np.ones((1, 5), np.uint8)  
# clean_lines = cv2.erode(image, kernel_line)
# cv2.imwrite('image.jpg', clean_lines)




