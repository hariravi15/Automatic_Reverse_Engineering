import cv2
import numpy as np

img = cv2.imread(r'D:\pic\mes2\top.png')           # <-- your file name
if img is None:
    raise IOError('image not found')

g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- your current pipeline ---
_, bw = cv2.threshold(g, 250, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# 1 px lines → thicker lines so close can bridge them
bw = cv2.dilate(bw, kernel, iterations=2)   # fatten the rings
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=3)
bw = cv2.erode(bw, kernel, iterations=2)    # shrink back to original width

# ----- do we finally have black holes inside the white disk? -----
temp = bw.copy()
cnts, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    outer = max(cnts, key=cv2.contourArea)
    cv2.drawContours(temp, [outer], -1, 128, -1)   # paint disk grey
cv2.imshow('4  after dilate+close+erode', temp)
print('values inside disk:', np.unique(temp))

# keep largest blob only
cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    outer = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(bw)
    cv2.drawContours(mask, [outer], -1, 255, -1)
    bw = cv2.bitwise_and(bw, mask)

# flood-fill background
holes = bw.copy()
cv2.floodFill(holes, None, (0,0), 255)
holes = cv2.bitwise_not(holes)
bw = cv2.bitwise_or(bw, holes)

# --- show ---
cv2.imshow('1  grey', g)
cv2.imshow('2  after close + keep largest', bw)
cv2.imshow('3  final mask (holes filled)', bw)
print('unique values in final mask:', np.unique(bw))

cv2.waitKey(0)
