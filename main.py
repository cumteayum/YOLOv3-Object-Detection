import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320
conf_threshold = 0.5
nms_threshold = 0.5

classes = []
with open("coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').rsplit('\n')
# print(classes)
# print(len(classes))

model_cfg = "/home/kali/IdeaProjects/yolo_detection/yolo-tiny.cfg"
model_weights = "/home/kali/IdeaProjects/yolo_detection/yolov3-tiny.weights"

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def find_objects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    class_ids = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - wT), int((det[1] * hT) - hT)
                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(float(confidence))

    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classes[class_ids[i]].upper()} {int(confs[i]*100)}%',
                    (x,y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (255,0,255), 2)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    # print(layer_names)
    output_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(output_names)

    outputs = net.forward(output_names)
    # print(type(outputs))

    find_objects(outputs, img)
    cv2.imshow("img", img)
    cv2.waitKey(1)
