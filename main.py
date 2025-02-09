from yolov5 import YOLOv5
import cv2
import numpy as np
import time
import send_email

model = YOLOv5("yolov5s.pt")
cam = cv2.VideoCapture(0)

overcrowding_threshold =1
last_sms_time = 0
sms_cooldown = 30
grid_size = 3  

while True:
    ret, frame = cam.read()
    if not ret:
        break

    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    results = model.predict(frame)

    people_count = 0
    grid_density = np.zeros((grid_size, grid_size), dtype=int)
        #here are predictions of persons
    for det in results.pred[0]:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if int(cls) == 0:  
            people_count += 1
            heatmap[int(y1):int(y2), int(x1):int(x2)] += conf

            # bound the grid that person has and update the density of that grid
            grid_x1, grid_y1 = int(x1 / frame.shape[1] * grid_size), int(y1 / frame.shape[0] * grid_size)
            grid_x2, grid_y2 = int(x2 / frame.shape[1] * grid_size), int(y2 / frame.shape[0] * grid_size)
            grid_density[grid_y1:grid_y2 + 1, grid_x1:grid_x2 + 1] += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    heatmap = np.clip(heatmap, 0, 255)
    heatmap = np.uint8(heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    combined_frame = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

    # overcrowd zone highlighted here
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_density[i, j] > overcrowding_threshold:  # crowd finded
                top_left = (j * frame.shape[1] // grid_size, i * frame.shape[0] // grid_size)
                bottom_right = ((j + 1) * frame.shape[1] // grid_size, (i + 1) * frame.shape[0] // grid_size)
                cv2.rectangle(combined_frame, top_left, bottom_right, (0, 0, 255), 2)

    # low crowd  so bound blue coloe 
    low_density_path = np.unravel_index(np.argsort(grid_density, axis=None), grid_density.shape)
    for point in zip(*low_density_path):
        if grid_density[point] == 0: 
            top_left = (point[1] * frame.shape[1] // grid_size, point[0] * frame.shape[0] // grid_size)
            bottom_right = ((point[1] + 1) * frame.shape[1] // grid_size, (point[0] + 1) * frame.shape[0] // grid_size)
            cv2.rectangle(combined_frame, top_left, bottom_right, (255, 255, 0), 2)
            cv2.putText(combined_frame, "Low", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.putText(combined_frame, f"People Count: {people_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Crowd Management", combined_frame)

    current_time = time.time()
    if people_count > overcrowding_threshold and current_time - last_sms_time > sms_cooldown:
        send_email.sendmail(
            "vivoharsha139@gmail.com", "717822p203@kce.ac.in", "fcktqlncwxtgwaik",
            "Over Crowding", "Over Crowding happening Stop it"
        )
        last_sms_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()