import cv2
# import supervision as sv
import numpy as np
from ultralytics import YOLO

model = YOLO('/content/runs/detect/train/weights/best.pt')





# For video

# for frame in sv.get_video_frames_generator(source_path='source_video.mp4'):

# while True:
#     ret, frame = cap.read()



# For image
# frame = cv2.imread('/content/datasets/img9-2/test/images/00000PORTRAIT_00000_BURST20220825113011337_jpg.rf.14586023dfde65f8746e52fce7d1ede0.jpg')

frame = cv2.imread('/content/Jin_Test_Images/4.jpeg')





results = model(frame, agnostic_nms=True)[0]
# results = model(frame)




# if not results or len(results) == 0:
# continue




for result in results:

  detection_count = result.boxes.shape[0]

  for i in range(detection_count):
    cls = int(result.boxes.cls[i].item())
    # name = result.names[cls]
    confidence = float(result.boxes.conf[i].item())
    bounding_box = result.boxes.xyxy[i].cpu().numpy()

    x = int(bounding_box[0])
    y = int(bounding_box[1])
    w = int(bounding_box[2] - x)
    h = int(bounding_box[3] - y)




    # print(cls)
    # # print(name)
    # print(confidence)
    # print(bounding_box)
    # print("========================")
    # print(x)
    # print(y)
    # print(w)
    # print(h)
    # print("=======================================================")







    # Coordinates and dimensions of the circular ROI
    center = (x + w // 2, y + h // 2)  # Center of the rectangle
    radius = min(w, h) // 2  # Radius is half of the minimum dimension

    # Create a circular mask
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)

    # Extract the circular ROI using the mask
    roi = cv2.bitwise_and(frame, mask)

    # Apply Gaussian blur to the circular ROI
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

    # Replace the original circular region in the image with the blurred ROI
    frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))  # Erase the original circular region
    frame = cv2.add(frame, blurred_roi)  # Add the blurred circular ROI









    # # Extract the region of interest (ROI) from the image based on the xywh coordinates:
    # roi = frame[y:y+h, x:x+w]

    # # Apply the blur to the ROI using a Gaussian blur. You can adjust the size of the Gaussian kernel (second argument) to control the level of blurring.
    # blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

    # # Replace the original region in the image with the blurred ROI:
    # frame[y:y+h, x:x+w] = blurred_roi




    # # For head
    # # Extract the region of interest (ROI) from the image based on the xywh coordinates:
    # roi = frame[int(y+0.2*h):y+h, x:x+w]

    # # Apply the blur to the ROI using a Gaussian blur. You can adjust the size of the Gaussian kernel (second argument) to control the level of blurring.
    # blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

    # # Replace the original region in the image with the blurred ROI:
    # frame[int(y+0.2*h):y+h, x:x+w] = blurred_roi




    # # For helmet
    # # Extract the region of interest (ROI) from the image based on the xywh coordinates:
    # roi = frame[int(y+0.5*h):y+h, x:x+w]

    # # Apply the blur to the ROI using a Gaussian blur. You can adjust the size of the Gaussian kernel (second argument) to control the level of blurring.
    # blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

    # # Replace the original region in the image with the blurred ROI:
    # frame[int(y+0.5*h):y+h, x:x+w] = blurred_roi





# Display
cv2.imwrite('./blurred_image.jpg', frame)

%cd {HOME}
Image(filename=f'{HOME}/blurred_image.jpg', width=1000)

     