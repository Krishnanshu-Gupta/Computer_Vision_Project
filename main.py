import cv2
import numpy as np

# Load image and convert to grayscale
image = cv2.imread('lots2.png')
grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply gaussian blur to reduce noise
blurred = cv2.GaussianBlur(grayed, (3, 3), 0)

# apply Otsu's threshold
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

edges = cv2.Canny(blurred, 200, 255)
edges2 = cv2.GaussianBlur(edges, (5, 5), 0)

# Calculate total area of the image
height, width = grayed.shape
total_area = width * height

# make sure countours are white, make sure they're in a roughly card aspect ratio
# try to get the numbers from the card

cnts = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
threshold_min_area = 400
white_threshold = 0.70
contours = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > threshold_min_area:
        contours += 1
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # calculate aspect ratio of the minimum bounding rectangle
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])
        aspect_ratio = width / height if width > height else height / width

        # Mask for the contour area
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [c], 0, 255, -1)

        white_pixels = cv2.countNonZero(cv2.bitwise_and(mask, thresh))

        # Calculate percentage of white pixels within the contour area
        white_pixel_percentage = white_pixels / area

        print(contours)
        print(aspect_ratio)
        print(width * height)
        print(white_pixel_percentage)
        print("")

        if white_pixel_percentage >= white_threshold:
            # draw contour
            cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
            #contours += 1

            # get contour centroid
            M = cv2.moments(c)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            cv2.putText(image, str(contours), (centroid_x, centroid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)

print("Contours detected:", contours)
cv2.imshow('thresh', thresh)
cv2.imshow("edges", edges2)
cv2.imshow('image', image)
cv2.waitKey()