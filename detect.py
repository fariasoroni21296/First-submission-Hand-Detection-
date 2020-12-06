import os
import cv2
import numpy as np
from argparse import ArgumentParser
from camera_config.camera import CameraStream

import torch
from model import MNIST_NET
from torch import optim
from torchvision import transforms
from PIL import Image
import traceback
# adding command line arguments
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)

ap = ArgumentParser()

ap.add_argument('--prototxt', '-p', type = str, help = 'please provide prototxt file with prototxt extension', default='hand/pose_deploy.prototxt')
ap.add_argument('--weight_file', '-w', type = str, help = 'please provide weight file with caffemodel extension', default='hand/pose_iter_102000.caffemodel')
ap.add_argument('--video_source', '-s', type = str, help='provide video source')#, default='.test.mp4'
ap.add_argument('--camera_source', '-c', type = int, help='provide camera source')
ap.add_argument('--confidence', '-t', type=float, help='minimum confidence threshold for model', default=0.2)
ap.add_argument('--write_video', '-wr', type = bool, help='boolean value whether the analyzed frames wil get written into disk or not. default True', default=True)

args = vars(ap.parse_args())

def getROI(canvas):
    # gray = cv2.bitwise_not(canvas)
    gray = canvas
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    # _, ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for i in range(len(ctrs)):
        x, y, w, h = cv2.boundingRect(ctrs[i])
        areas.append((w*h,i))

    def sortSecond(val): 
        return val[0]  
        
    areas.sort(key = sortSecond,reverse = True)
    x, y, w, h = cv2.boundingRect(ctrs[areas[1][1]])
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 1)
    roi = gray[y:y + h, x:x + w]
    #cv2.imshow('ROI', roi)
    return roi

def sequential_model():
    import torch.nn as nn
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
    
    return model

# defining constants
N_POINTS = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],
                [9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
THRESHOLD = args['confidence']

# defining capturing pointer from openCV

if args['video_source']:
    try:
        cap = cv2.VideoCapture(args['video_source'])
        ret, frame = cap.read()
    except Exception as e:
        print(e)
else:
    try:
        cap = CameraStream(args['camera_source'], width=512, height=512).start()
        ret, frame = cap.read()
    except Exception as e:
        print(e)

frame_width, frame_height = frame.shape[1], frame.shape[0]

# defining writer if True

if args['write_video']:
    writing_path = os.path.join(os.getcwd(), 'demo')
    writer = cv2.VideoWriter(writing_path+'/analyzed_output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))

# setting model with cv2.dnn module

net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['weight_file'])

#defining drawing canvas
canvas = np.zeros((frame.shape[1], frame.shape[1]), np.uint8)
far_points = []
drawing = False
prev_index_finger_tip = None
# model = MNIST_NET()
model = sequential_model()
model.to(device)
model_state = torch.load('./results/sequential_model.pth', map_location=device)
model.load_state_dict(model_state)
# model.to(device)
print("MODEL\n", model)
# optimizer = optim.SGD(model.parameters(), lr = )

while ret:
    try:
        copied_frame = frame.copy()
        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas[:,:] = 0
        image_blob = cv2.dnn.blobFromImage(im_rgb, 1.0 / 255, (368, 368), (0, 0, 0), swapRB = True, crop = False)

        net.setInput(image_blob)

        output = net.forward()

        # to store detected key points
        points_list = []
        key = cv2.waitKey(1)

        for i in range(N_POINTS):
            # confidence map of corresponding body's part.
            prob_map = output[0, i, :, :]
            prob_map = cv2.resize(prob_map, (frame_width, frame_height))

            # Find global maxima of the prob_map.
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            if prob > THRESHOLD :
                # cv2.circle(copied_frame, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(copied_frame, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points_list.append((int(point[0]), int(point[1])))
            else :
                points_list.append(None)
        if key & 0xFF == ord('d'):
            drawing = True
        if key & 0xFF == ord('c'):
            drawing = False
            canvas[:, :] =  0
            far_points.clear()
        
        index_finger_tip = False

        # drawing the skeleton
        for pair in POSE_PAIRS:
            part_A = pair[0]
            part_B = pair[1]

            if points_list[part_A] and points_list[part_B]:
                if points_list[8]:
                    prev_index_finger_tip = points_list[8]
                cv2.line(frame, points_list[part_A], points_list[part_B], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, points_list[part_A], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points_list[part_B], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            elif points_list[8] is None:
                cv2.circle(frame, prev_index_finger_tip, 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        
        if drawing:
            if len(far_points)>100:
                far_points.pop(0)
            if points_list[8] is None:
                far_points.append(prev_index_finger_tip)
            else:
                far_points.append(points_list[8])
            for i in range(len(far_points)-1):
                cv2.line(frame, far_points[i], far_points[i+1], (255,5,255), 20)
                cv2.line(canvas, far_points[i], far_points[i+1], (255,255,255), 20)

        if key & 0xFF == ord('s'):
            drawing = False
            # img = cv2.resize(canvas, (28, 28))
            canvas = cv2.flip(canvas, 1)
            # roi = getROI(canvas)
            # img = cv2.resize(roi, (28, 28))
            img = cv2.resize(canvas, (28, 28))
            img = cv2.GaussianBlur(img,(5,5),0)
            cv2.imshow("ROI", img)
            img = Image.fromarray(img)

            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

            processed_img = preprocess(img)
            # processed_img = processed_img.reshape([1, 1, 28, 28]).float().to(device)
            # processed_img = processed_img.to(device)
            processed_img = processed_img.view(1, 784).to(device)
            with torch.no_grad():
                log_prediction = model(processed_img)#torch.transpose(processed_img, 2, 3)
                # log_prediction = model(processed_img)
                print(log_prediction)
                prob = torch.exp(log_prediction)
                prob = list(prob.cpu().numpy()[0])
                print("Predicted digit = ", prob.index(max(prob)))
                # _, prediction = torch.max(log_prediction, 1)
                # print("Probability: ", prob)
                # preds = np.squeeze(prediction.detach())
                # print("Predicted Digit =", np.amax(preds))
        # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, "Hand Pose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Output-Skeleton', frame)
        cv2.imshow('Canvas', canvas)

        if key & 0xFF == ord('q'):
            break

        if args['write_video']:
            writer.write(frame)
        
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
    except Exception as e:
        print("Problem Found! : ", e)
        ret, frame = cap.read()
        cv2.imshow('Output-Skeleton', frame)
        traceback.print_exc()
        pass
    
if args['video_source']:
    cap.release()
else:
    cap.stop()
if args['write_video']:
    writer.release()

# transforms.Resize((28,28)),