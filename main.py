from ultralytics import YOLO
import cv2
import csv  # Import the csv module

# Paths configuration
csv_file_path = '/Users/ishaangupta/Downloads/main/assderp/jorgensen-v5/SharkBMI/main.csv'
image_path = '/Users/ishaangupta/Downloads/main/assderp/jorgensen-v5/pose-detection-keypoints-estimation-yolov8/data/images/val/AN15093001_14ft_MALE_02frame371.jpg'

# Model paths
model1_path = '/Users/ishaangupta/Downloads/main/assderp/jorgensen-v5/SharkBMI/OcculationValidv2/weights/last.pt'
model2_path = '/Users/ishaangupta/Downloads/main/assderp/jorgensen-v5/SharkBMI/PartialValid/weights/last.pt'

# Load the image
img = cv2.imread(image_path)

# Initialize both models
model1 = YOLO(model1_path)
model2 = YOLO(model2_path)  # Assuming the second model can be initialized similarly

# Process the image with both models
results1 = model1(image_path)[0]
results2 = model2(image_path)[0]

# Open the CSV file in append mode
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Process results from the first model
    for result in results1:
        for keypoint_index, keypoint in enumerate(result.keypoints.data.tolist()):
            for i in range(4):  # Assuming 4 keypoints for the first model
                writer.writerow(['model1', keypoint_index, keypoint[i][0], keypoint[i][1]])
                cv2.putText(img, f"M1-{keypoint_index}", (int(keypoint[i][0]), int(keypoint[i][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Process results from the second model
    for result in results2:
        for keypoint_index, keypoint in enumerate(result.keypoints.data.tolist()):
            # This time, only process the third point from Model 2
            for i in range(4): #this is just very scuffed line to make the last point record and print only :(
                if i == 1:  # Check if it's the third keypoint
                    # Since we're only interested in the third point, no need to loop through 'i'
                    writer.writerow(['model2', keypoint_index, keypoint[4][0], keypoint[4][1]])
                    cv2.putText(img, f"M2-{keypoint_index}", (int(keypoint[4][0]), int(keypoint[4][1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the image with keypoints from both models
cv2.imshow('Keypoints from Two Models', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
