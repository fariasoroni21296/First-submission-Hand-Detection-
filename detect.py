import os
import cv2
import numpy as np
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

while ret:
    try:
        copied_frame = frame.copy()
        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_blob = cv2.dnn.blobFromImage(im_rgb, 1.0 / 255, (368, 368), (0, 0, 0), swapRB = True, crop = False)

        net.setInput(image_blob)

        output = net.forward()

        # to store detected key points
        points_list = []

        for i in range(N_POINTS):
            # confidence map of corresponding body's part.
            prob_map = output[0, i, :, :]
            prob_map = cv2.resize(prob_map, (frame_width, frame_height))

            # Find global maxima of the prob_map.
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            if prob > THRESHOLD :
                cv2.circle(copied_frame, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(copied_frame, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points_list.append((int(point[0]), int(point[1])))
            else :
                points_list.append(None)

        # drawing the skeleton
        for pair in POSE_PAIRS:
            part_A = pair[0]
            part_B = pair[1]

            if points_list[part_A] and points_list[part_B]:
                cv2.line(frame, points_list[part_A], points_list[part_B], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, points_list[part_A], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points_list[part_B], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


        # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, "Hand Pose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Output-Skeleton', frame)

        if cv2.waitKey(1) == 0xFF & ord('q'):
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