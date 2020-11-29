import os
import cv2
import numpy as np
# from imutils import grab_contours
from argparse import ArgumentParser
from camera_config.camera import CameraStream

# adding command line arguments

ap = ArgumentParser()

ap.add_argument('--prototxt', '-p', type = str, help = 'please provide prototxt file with prototxt extension', default='hand/pose_deploy.prototxt')
ap.add_argument('--weight_file', '-w', type = str, help = 'please provide weight file with caffemodel extension', default='hand/pose_iter_102000.caffemodel')
ap.add_argument('--video_source', '-s', type = str, help='provide video source')#, default='.test.mp4'
ap.add_argument('--camera_source', '-c', type = int, help='provide camera source')
ap.add_argument('--confidence', '-t', type=float, help='minimum confidence threshold for model', default=0.2)
ap.add_argument('--write_video', '-wr', type = bool, help='boolean value whether the analyzed frames wil get written into disk or not. default True', default=True)

args = vars(ap.parse_args())

# defining constants
# N_POINTS = 22
N_POINTS = 10#[0,5,6,7,8]
# POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],
#                 [9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
POSE_PAIRS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8]]
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
        cap = CameraStream(args['camera_source'], width=640, height=480).start()
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

finger_tip_list = []
drawing_track_list = []
drawing = False
while ret:
    try:
        resized_frame = cv2.resize(frame, (600, 600))
        im_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image_blob = cv2.dnn.blobFromImage(im_rgb, 1.0 / 255, (368, 368), (0, 0, 0), swapRB = True, crop = False)

        net.setInput(image_blob)

        output = net.forward()

        # to store detected key points
        points_list = []

        # for i in range(N_POINTS):
        for i in range(N_POINTS):
            # confidence map of corresponding body's part.
            prob_map = output[0, i, :, :]
            prob_map = cv2.resize(prob_map, (frame_width, frame_height))

            # Find global maxima of the prob_map.
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            if prob > THRESHOLD :
                # Add the point to the list if the probability is greater than the threshold
                points_list.append((int(point[0]), int(point[1])))
            else :
                points_list.append(None)

        # drawing the skeleton
        for pair in POSE_PAIRS:
            part_A = pair[0]
            part_B = pair[1]

            if points_list[part_A] and points_list[part_B]:
                if points_list[7] and points_list[8]:
                    if drawing:
                        finger_tip_list.append(points_list[8])
                        drawing_track_list.append(0)
                    cv2.circle(frame, points_list[8], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                else:
                    finger_tip_list.clear()
        
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('d'):
            drawing = True
        if drawing:    
            for i in range(len(finger_tip_list)):
                if drawing_track_list[i] == 0:
                    cv2.line(frame, finger_tip_list[i-1], finger_tip_list[i], (0, 0, 255), 3)
                    drawing_track_list[i] = 1
                else:
                    continue
                if key & 0xFF == ord('c'):
                    drawing = False
                    finger_tip_list.clear()
                    break


        # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, "Hand Pose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Output-Skeleton', frame)

        if key == 0xFF & ord('q'):
            break

        if args['write_video']:
            writer.write(frame)
        
        ret, frame = cap.read()
    except Exception as e:
        print("Problem Found! : ", e)
        ret, frame = cap.read()
        cv2.imshow('Output-Skeleton', frame)
        pass
    
if args['video_source']:
    cap.release()
else:
    cap.stop()
if args['write_video']:
    writer.release()