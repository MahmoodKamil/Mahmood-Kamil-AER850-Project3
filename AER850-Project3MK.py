'''Step 1: Object Masking'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = 'C:/Users/mahmo/Downloads/motherboard_image (1).JPEG'
img_real = cv2.imread(path, cv2.IMREAD_COLOR)
img_real = cv2.rotate(img_real, cv2.ROTATE_90_CLOCKWISE)

img = cv2.imread(path, cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(img, (47, 47), 4)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

edges = cv2.Canny(img_gray, 50, 300)
edges = cv2.dilate(edges, None, iterations=10)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(img_real)
cv2.drawContours(image=mask, contours=[max(contours, key=cv2.contourArea)], contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)

masked_img = cv2.bitwise_and(mask, img_real)

edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
original_rgb = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)

# Plots
plt.figure(figsize=(20, 5))
plt.subplot(142)
plt.imshow(edges_rgb)
plt.title('Edge Detection')
plt.axis('off')

plt.subplot(143)
plt.imshow(mask_rgb)
plt.title('Mask')
plt.axis('off')

plt.subplot(144)
plt.imshow(masked_img_rgb)
plt.title('Final Extracted PCB')
plt.axis('off')

plt.tight_layout()
plt.show()



'''Step 2: YOLOv8 Training'''
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='C:/Users/mahmo/Downloads/Project 3 Data/data/data.yaml', epochs=170, batch=2, imgsz=1024, name='pcb_identifier') 



'''Step 2.5: Matrices and Graph Curves'''
import matplotlib.image as mpimg

# Normalized confusion matrix
conf_matrix = mpimg.imread('/content/runs/detect/pcb_identifier/confusion_matrix_normalized.png')
plt.figure(figsize=(12, 8))
plt.imshow(conf_matrix)
plt.axis('off')
plt.show()

# Precision-Confidence curve
pc_curve = mpimg.imread('/content/runs/detect/pcb_identifier/P_curve.png')
plt.figure(figsize=(12, 8))
plt.imshow(pc_curve)
plt.axis('off')
plt.show()

# Precision-Recall curve
pr_curve = mpimg.imread('/content/runs/detect/pcb_identifier/PR_curve.png')
plt.figure(figsize=(12, 8))
plt.imshow(pr_curve)
plt.axis('off')
plt.show()


'''Step 3: YOLOv8 Evaluation'''
from google.colab.patches import cv2_imshow

model = YOLO('/content/runs/detect/pcb_identifier/weights/best.pt')

images1 = ['C:/Users/mahmo/Downloads/Project 3 Data/data/evaluation/ardmega.jpg']

for img in images1:
    print(f"\nResults for: {img.split('/')[-1]}")
    results = model.predict(img, imgsz=1024, conf=0.25)
    cv2_imshow(results[0].plot())
     
    
    model = YOLO('/content/runs/detect/pcb_identifier/weights/best.pt')

images2 = ['C:/Users/mahmo/Downloads/Project 3 Data/data/evaluation/arduno.jpg']

for img in images2:
    print(f"\nResults for: {img.split('/')[-1]}")
    results = model.predict(img, imgsz=1024, conf=0.25)
    cv2_imshow(results[0].plot())
    
    
    
    model = YOLO('/content/runs/detect/pcb_identifier/weights/best.pt')

images3 = ['C:/Users/mahmo/Downloads/Project 3 Data/data/evaluation/rasppi.jpg']

for img in images3:
    print(f"\nResults for: {img.split('/')[-1]}")
    results = model.predict(img, imgsz=1024, conf=0.25)
    cv2_imshow(results[0].plot())