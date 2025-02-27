import math
import os
import threading
from copy import copy
import pickle

import requests
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
    time_synchronized, TracedModel
from utils.download_weights import download

# For SORT tracking
import skimage
from sort import *

import http.server
import socketserver
import shutil

from io import BytesIO
from PIL import Image
import io

crosswalk_timer = 0
crossing_timer = 0


def send_green():
    #requests.get("http://<Arduino IP>/green/on")
    pass


def send_red():
    #requests.get("http://<Arduino IP>/red/on")
    pass


class ImageHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        # Get the content length of the incoming request
        content_length = int(self.headers['Content-Length'])

        # Read the incoming image data from the request body
        image_data = self.rfile.read(content_length)
        image = Image.open(io.BytesIO(image_data))
        image.show()

        # Write the image data to a file
        # image.save("test.jpg")

        # Send a response back to the client
        self.send_response(200)


# ............................... Bounding Boxes Drawing ............................
"""Function to Draw Bounding boxes"""


def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None,
               offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id) + ":" + names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0]) / img.shape[1], int(box[1]) / img.shape[0], int(box[2]) / img.shape[1],
                int(box[3]) / img.shape[0], int(box[0] + (box[2] * 0.5)) / img.shape[1],
                int(box[1] + (
                        box[3] * 0.5)) / img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img


# ..............................................................................


def detect(opt, save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # .... Initialize SORT ....
    # .........................
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    # .........................

    # ........Rand Color for every trk.......
    rand_color_list = []
    for i in range(0, 5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    # ......................................

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True,
                                                                                 exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    # para = 17  # FIXME clear this eventually

    people_stopped = 0

    # if opt.target == "pedestrians":
    #     shared = {"cars": 0}
    #     fp = open("shared.pkl", "r")
    #     pickle.dump(shared, fp)
    # else:
    #     fp = open("shared.pkl", "w")
    #
    # pickle.load(fp)
    car_on_traffic_light = 0
    go_green = False

    for path, img, im0s, vid_cap in dataset:  # for each frame
        people_stopped = 0
        car_on_traffic_light = 0
        # Wait for the user to press a key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # FIXME
        # para -= 1
        # if para == 0:
        #     para = 17
        # else:
        #     continue
        # FIXME END

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Get the center coordinates of the image
                h, w = 640, 960
                cx, cy = w // 2 + int(opt.box_centerx), h // 2 + int(opt.box_centery)
                ccarx, ccary = w // 2 + int(opt.carbox_centerx), h // 2 + int(opt.carbox_centery)

                # Set the width and height of the box
                w_box, h_box = 100 + int(opt.box_sizex), 100 + int(opt.box_sizey)

                xcar1, ycar1 = (ccarx - w_box // 2), (ccary - h_box // 2)
                xcar2, ycar2 = (ccarx + w_box // 2), (ccary + h_box // 2)

                # Calculate the coordinates of the top-left and bottom-right corners of the box
                x1, y1 = (cx - w_box // 2), (cy - h_box // 2)
                x2, y2 = (cx + w_box // 2), (cy + h_box // 2)
                # Write results
                want_to_cross = 0

                # ..................USE TRACK FUNCTION....................
                # pass an empty array to sort
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                for cx1, cy1, cx2, cy2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([cx1, cy1, cx2, cy2, conf, detclass])))

                    if detclass == 0:  # if person is in frame

                        # (cx1, cy1), (cx2, cy2) = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                        pcx, pcy = int((cx1 + cx2) // 2), int((cy1 + cy2) // 2)
                        cv2.circle(im0, (pcx, pcy), 5, (0, 0, 255), -1)

                        cv2.circle(im0, (x1, y1), 5, (0, 255, 255), -1)
                        cv2.circle(im0, (x2, y2), 5, (255, 255, 0), -1)
                        # Draw the box
                        # if x1 <= pcx <= x2 and y1 <= pcy <= y2:
                        #     # print("someone wants to crosss :))))))))))))))))))")
                        #     want_to_cross += 1

                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                txt_str = ""

                # loop over tracks
                for track in tracks:
                    # print()
                    # print(track.id, track.centroidarr)
                    # print()
                    size = len(track.centroidarr)

                    # if car
                    # if track.detclass == 2:
                    #     if size > 2:  # TODO dependent on resolution being 960 x 640
                    #         if x1 <= track.centroidarr[size - 1][0] <= x2 and y1 <= track.centroidarr[size - 1][
                    #             1] <= y2:
                    #             car_on_traffic_light += 1
                    # if car stopped
                    # if math.sqrt((track.centroidarr[size - 2][0] - track.centroidarr[size - 1][0]) ** 2 + (
                    #         track.centroidarr[size - 2][1] - track.centroidarr[size - 1][1]) ** 2) < 10:
                    #     people_stopped += 1

                    # if person
                    if track.detclass == 0:
                        if size > 2:  # TODO dependent on resolution being 960 x 640
                            if xcar1 <= track.centroidarr[size - 1][0] <= xcar2 and ycar1 <= \
                                    track.centroidarr[size - 1][1] <= ycar2:
                                if math.sqrt((track.centroidarr[size - 2][0] - track.centroidarr[size - 1][0]) ** 2 + (
                                        track.centroidarr[size - 2][1] - track.centroidarr[size - 1][1]) ** 2) < 5:
                                    car_on_traffic_light += 1
                            if x1 <= track.centroidarr[size - 1][0] <= x2 and y1 <= track.centroidarr[size - 1][
                                1] <= y2:
                                # if person stopped
                                if math.sqrt((track.centroidarr[size - 2][0] - track.centroidarr[size - 1][0]) ** 2 + (
                                        track.centroidarr[size - 2][1] - track.centroidarr[size - 1][1]) ** 2) < 5:
                                    people_stopped += 1

                    # color = compute_color_for_labels(id)
                    # draw colored tracks
                    if colored_trk:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                        int(track.centroidarr[i][1])),
                                  (int(track.centroidarr[i + 1][0]),
                                   int(track.centroidarr[i + 1][1])),
                                  rand_color_list[track.id], thickness=2)
                         for i in range(0, len(track.centroidarr))
                         if i < len(track.centroidarr) - 1]
                    # draw same color tracks
                    else:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                        int(track.centroidarr[i][1])),
                                  (int(track.centroidarr[i + 1][0]),
                                   int(track.centroidarr[i + 1][1])),
                                  (255, 0, 0), thickness=2)
                         for i in range(0, len(track.centroidarr))
                         if i < len(track.centroidarr) - 1]

                    if save_txt and not save_with_object_id:
                        # Normalize coordinates
                        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1],
                                                    track.centroidarr[-1][1] / im0.shape[0])
                        if save_bbox_dim:
                            txt_str += " %f %f" % (
                                np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0],
                                np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                        txt_str += "\n"

                cv2.putText(im0, str(people_stopped), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3,
                            cv2.LINE_AA,
                            False)
                cv2.putText(im0, str(car_on_traffic_light), (h - 70 * 2, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 255), 3,
                            cv2.LINE_AA,
                            False)

                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)

                if car_on_traffic_light > 0:
                    cv2.rectangle(im0, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 2)
                elif car_on_traffic_light == 0:
                    cv2.rectangle(im0, (xcar1, ycar1), (xcar2, ycar2), (0, 0, 255), 2)

                if people_stopped > 0:
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif people_stopped == 0:
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # draw boxes for visualization
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path)
                # ........................................................

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            current_n_cars = car_on_traffic_light
            global crosswalk_timer
            global crossing_timer

            if people_stopped == 0:
                crosswalk_timer = time.time()
                cv2.circle(im0, (30, 30), 10, (0, 0, 255), -1)
                send_red()
            elif people_stopped >= current_n_cars:
                if (time.time() - crosswalk_timer) >= 2:
                    green = True
                    crossing_timer = time.time()
                    cv2.circle(im0, (30, 30), 10, (0, 255, 0), -1)
                    send_green()
                    go_green = True
                else:
                    cv2.circle(im0, (30, 30), 10, (0, 0, 255), -1)
            else:
                crosswalk_timer = time.time()
                cv2.circle(im0, (30, 30), 10, (0, 0, 255), -1)
                send_red()

            # Stream results
            if True:  # FIXME revert me to view_img
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    
                    # print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

            if go_green:
                time.sleep(5)
                go_green = False

    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',
                        help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true',
                        help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    parser.add_argument('--box_centerx', dest='box_centerx', default='0', help='box_centerx')
    parser.add_argument('--box_centery', dest='box_centery', default='0', help='box_centery')
    parser.add_argument('--box_sizex', dest='box_sizex', default='0', help='box_sizex')
    parser.add_argument('--box_sizey', dest='box_sizey', default='0', help='box_sizey')
    parser.add_argument('--ip', dest='ip', default='', help='ip address')
    parser.add_argument('--target', dest='target', default='pedestrians', help='pedestrians or cars')

    parser.set_defaults(download=True)
    opt = parser.parse_args()
    # check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(str(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    # PORT = 8002
    #
    # with socketserver.TCPServer(("", PORT), ImageHandler) as httpd:
    #     print("Serving on port", PORT)
    #     httpd.serve_forever()

    with torch.no_grad():
        # python detect_and_track.py --nosave --box_centerx 0 --box_centery 100 --box_sizex 760 --box_sizey 150 --classes 0 --source P1210189.MP4
        if opt.target == "pedestrians":
            opt.classes = 0
            opt.source = "pedestrians.MP4"
            opt.box_centerx = -300
            opt.box_centery = 0
            opt.carbox_centerx = 300
            opt.carbox_centery = 0
            opt.box_sizex = 460
            opt.box_sizey = 600

        # python detect_and_track.py --nosave --box_centerx 0 --box_centery 270 --box_sizex 980 --box_sizey 350 --classes 2 --source PXL_20230322_185218149.mp4
        else:
            opt.classes = 2
            opt.source = "cars.mp4"
            opt.box_centerx = 0
            opt.box_centery = 0
            opt.box_sizex = 980
            opt.box_sizey = 350
        if opt.ip:
            opt.source = "http://{}:8079/video_feed".format(opt.ip)

        detect(opt)

