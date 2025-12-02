import cv2
import cvzone

Thres = 0.8

# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects list")
print(classes)


# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
while True:
    # Get frames
    ret, frame = cap.read()

    # Object Detection
    (class_ids, confs, bboxes) = model.detect(frame, confThreshold=Thres)
    for class_id, confidence, bbox in zip(class_ids, confs, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (220, 0, 50), 2)
        cv2.putText(frame, str(round(confidence * 100, 2)), [x, y - 25], cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
        cvzone.cornerRect(frame, (x, y, w, h))

    print("class_ids", class_ids)
    print("confs", confs)
    print("bboxes", bboxes)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
