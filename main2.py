from ultralytics import YOLO
import cv2
import csv
import os  # Import the os module to interact with the filesystem

# Paths configuration
csv_file_path = '/Users/ishaangupta/Downloads/main/assderp/jorgensen-v5/SharkBMI/main.csv'
image_folder = '/Users/ishaangupta/Downloads/main/assderp/jorgensen-v5/SharkBMI/binaryClass/output/class_0'

# Model paths
model1_path = '/Users/ishaangupta/Downloads/main/assderp/jorgensen-v5/SharkBMI/OcculationValidv2/weights/last.pt'
model2_path = '/Users/ishaangupta/Downloads/main/assderp/jorgensen-v5/SharkBMI/PartialValid/weights/last.pt'

# Initialize both models
model1 = YOLO(model1_path)
model2 = YOLO(model2_path)

# Open the CSV file in append mode
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Loop through all images in the directory
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(('.jpg', '.png')):  # Check for image files
            image_path = os.path.join(image_folder, image_filename)
            img = cv2.imread(image_path)

            # Extract name from image_path
            image_name = image_filename.split('.')[0]  # Removes file extension

            # Process the image with both models
            results1 = model1(image_path)[0]
            results2 = model2(image_path)[0]

            # Gather all keypoints and bounding box data for model 1
            output1 = [image_name]
            for result in results1:
                for keypoint_index, keypoint in enumerate(result.keypoints.data.tolist()):
                    for i in range(4):  # Assuming 4 keypoints for the first model
                        output1.extend([keypoint[i][0], keypoint[i][1]])
                for box in result.boxes:
                    if box.conf[0].item() > 0.8:
                        try:
                            output1.extend([box.xyxy])
                        except:
                            print(image_name + " bounding box values not found")
                            output1.append("N/A")

            # Gather all keypoints and bounding box data for model 2
            for result in results2:
                for keypoint_index, keypoint in enumerate(result.keypoints.data.tolist()):
                    output1.extend([keypoint[4][0], keypoint[4][1]])
                for box in result.boxes:
                    if box.conf[0].item() > 0.8:
                        output1.append(box.conf[0].item())
                        writer.writerow(output1)
                        break  # To ensure only one line is written per image if viable
                    else:  # Executed if no box was viable for analysis
                        print(image_name + " not viable for analysis: " + str(box.conf[0].item()))
