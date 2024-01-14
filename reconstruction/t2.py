import cv2

clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point: {x}, {y}")

image = cv2.imread('/Users/keisaiki/Documents/Lab/3DReconstruction/reconstruction/projects/1/imames_resized/in1.jpg')

cv2.namedWindow('image')
cv2.setMouseCallback('image', click_event)

cv2.imshow('image', image)

cv2.waitKey(0)

cv2.destroyAllWindows()

print("Clicked Points: ", clicked_points)
