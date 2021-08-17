import numpy as np
import cv2
from matplotlib import pyplot as plt

coin_rad = 0
crack_width = 0

img_path = 'images\cracked_coin_04.jpg'

def getCoinRadius(img_path=''):
    img = cv2.imread(img_path)

    # image processing
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 1.5)

    def detect_circles(input_img):
        circles = cv2.HoughCircles(input_img, cv2.HOUGH_GRADIENT, 1, 700, param1=70, param2=50, minRadius=0, maxRadius=200)
        return np.uint16(np.around(circles))
            
    detected_circles = detect_circles(gray)

    # visualize results:
    count = 1
    for (x, y, r) in detected_circles[0, :]:
        cv2.circle(img, (x, y), r , (0,0,255), 3)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 5)
        cv2.line(img, (x,y), (x+r,y), (255,0,0),3)
        # displaying radius:
        cv2.putText(img,f"radius:12.5mm",(x,y), cv2.FONT_ITALIC, fontScale= 1, color = (255, 0, 0), thickness = 2)
    
        count += 1

    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    cv2.imshow('output',img)
    return [r,img]

def getCrackWidth(img_path=''):
    # read a cracked sample image
    img = cv2.imread(img_path)
    width = int(img.shape[1]//2)
    height = int(img.shape[0]//2)

    # Convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray white crack', cv2.resize(gray,(width,height)))
    # Image processing ( smoothing )
    # Averaging
    blur = cv2.blur(gray,(3,3))

    cv2.imshow('blur white crack', cv2.resize(blur,(width,height)))
    # Apply logarithmic transform
    img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255

    # Specify the data type
    img_log = np.array(img_log,dtype=np.uint8)

    # Image smoothing: bilateral filter
    # FINE TUNE THIRD PARAMETER FOR BETTER RESULTS
    bilateral = cv2.bilateralFilter(img_log, 5, 0, 75)

    cv2.imshow('bilateral white crack', cv2.resize(bilateral,(width,height)))
    # Canny Edge Detection
    edges = cv2.Canny(bilateral,100,200)

    # cv2.imshow('edge white crack', cv2.resize(edges,(int(img.shape[1]//2),int(img.shape[0]//2))))
    # Morphological Closing Operator
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('black white crack', cv2.resize(closing,(width,height)))
    # pixel width of wall cracks in every row
    img_row_sum = np.sum(closing==255,axis=1)

    # 90th percentile of wall crack widths
    crack_width = np.percentile(img_row_sum,90)
    print(crack_width)

    return crack_width

# cv2.imshow('image', getCoinRad(img_path)[1])
coin_rad = getCoinRadius(img_path)[0]
# crack_width = getCrackWidth(getCoinRad(img_path)[1])
crack_width = getCrackWidth(img_path)

# Classification
crack_actual_width = (crack_width/coin_rad) * 12.5
crack_status = ''
if crack_actual_width < 5:
    crack_status = 'SLIGHT'
elif crack_actual_width >= 5:
    crack_status = 'MODERATE'
elif crack_actual_width >= 25:
    crack_status = 'SEVERE'
print(coin_rad, crack_width, f'Actual Crack Width: {round(crack_actual_width,2)}mm / STATUS: {crack_status}')


# Displaying Final Result with Edited Image
img = cv2.imread(img_path)
img = cv2.putText(img, f'Actual Crack Width: {round(crack_actual_width,2)}mm / STATUS: {crack_status}', (0,img.shape[0]-20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 3)
def rescaleFrame(frame,scale=0.5):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)

    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)
img = rescaleFrame(img)
cv2.imshow('FINAL OUTPUT',img)

cv2.waitKey(0)