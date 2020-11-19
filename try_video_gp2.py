import os
from absl import app
from absl import flags
from absl import logging
#import matplotlib.pyplot as plt
import model
import numpy as np
import fnmatch
import tensorflow as tf
import nets
import util
import cv2

gfile = tf.gfile

#yolo from here-----
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
#------------------

#----------------------------------------------------------------------------------------------------------------------
#get bbox and color data

color_data = []

#from https://qiita.com/yasudadesu/items/dd3e74dcc7e8f72bc680

def drow_texts(img, x, y, texts, font_scale, color, thickness):
    initial_y = 0
    dy = int (img.shape[0] / 20)

    texts = [texts] if type(texts) == str else texts

    for i, text in enumerate(texts):
        offset_y = y + (i+1)*dy
        cv2.putText(img, text, (x, offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def color_detection(img, xy, color=None):
    harf_height = int(img.shape[0] / 2)
    #get box area
    boxFromX = int(xy[0])
    boxFromY = int(xy[1] + harf_height)
    boxToX = int(xy[2])
    boxToY = int(xy[3] + harf_height)

    #get box
    imgBox = img[boxFromY: boxToY, boxFromX:boxToX]

    #flatten and out RGB mean
    b = int(imgBox.T[0].flatten().mean())
    g = int(imgBox.T[1].flatten().mean())
    r = int(imgBox.T[2].flatten().mean())

    #wite color data on the depth
    color = color or [random.randint(0, 255) for _ in range(3)]
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) 
    text = ["B: %.2f" % (b), "G: %.2f" % (g), "R: %.2f" % (r)]
    drow_texts(img, boxFromX, boxFromY, text, 0.3, color, tl)


#write bbox on depth trying
def plot_bbox_and_depth(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    harf_ymax = int(img.shape[0] / 2)
    k1, k2 = (int(x[0]), int(x[1]) + harf_ymax), (int(x[2]), int(x[3]) + harf_ymax)
    cv2.rectangle(img, k1, k2, color, thickness=tl, lineType=cv2.LINE_AA)

#--------------------------------------yolo main--------------------------------------------

def detect(source, save_img=False):
  weights = opt.weights
  view_img = opt.view_img
  imgsz = 640

  # Directories
  #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
  #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

  # Initialize
  set_logging()
  device = select_device(opt.device)
  half = device.type != 'cpu'  # half precision only supported on CUDA

  # Load model
  model = attempt_load(weights, map_location=device)  # load FP32 model
  imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
  if half:
      model.half()  # to FP16

  # Second-stage classifier
  classify = False
  if classify:
      modelc = load_classifier(name='resnet101', n=2)  # initialize
      modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

  # Set Dataloader
  #vid_path, vid_writer = None, None
  #dataset = LoadStreams(source, img_size=imgsz)

  # Get names and colors
  names = model.module.names if hasattr(model, 'module') else model.names
  colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

  # Run inference
  img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
  _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
  img = torch.from_numpy(source).to(device)
  img = img.half() if half else img.float()  # uint8 to fp16/32
  img /= 255.0  # 0 - 255 to 0.0 - 1.0
  if img.ndimension() == 3:
      img = img.unsqueeze(0)

  # Inference
  pred = model(img, augment=opt.augment)[0]

  # Apply NMS
  pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

  # Apply Classifier
  if classify:
    pred = apply_classifier(pred, modelc, img, im0s)

  # Process detections
  for i, det in enumerate(pred):  # detections per image
    p, s, im0 = Path(path), '', im0s

    save_path = str(save_dir / p.name)
    txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
    s += '%gx%g ' % img.shape[2:]  # print string
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += '%g %ss, ' % (n, names[int(c)])  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            if save_img or view_img:  # Add bbox to image
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_bbox_and_depth(xyxy, im0, label=label, color=colors[int(cls)])
                color_detection(im0, xyxy, color=colors[int(cls)])


#--------------------------------------------------------------------------------------------------


def mask_image_stack(input_image_stack, input_seg_seq):
  background = [mask == 0 for mask in input_seg_seq]
  background = reduce(lambda m1, m2: m1 & m2, background)
  # If masks are RGB, assume all channels to be the same. Reduce to the first.
  if background.ndim == 3 and background.shape[2] > 1:
    background = np.expand_dims(background[:, :, 0], axis=2)
  elif background.ndim == 2:  # Expand.
    background = np.expand_dism(background, axis=2)
  # background is now of shape (H, W, 1).
  background_stack = np.tile(background, [1, 1, input_image_stack.shape[3]])
  return np.multiply(input_image_stack, background_stack)

def create_output_dirs(im_files, basepath_in, output_dir):
  """Creates required directories, and returns output dir for each file."""
  output_dirs = []
  for i in range(len(im_files)):
    relative_folder_in = os.path.relpath(
        os.path.dirname(im_files[i]), basepath_in)
    absolute_folder_out = os.path.join(output_dir, relative_folder_in)
    if not gfile.IsDirectory(absolute_folder_out):
      gfile.MakeDirs(absolute_folder_out)
    output_dirs.append(absolute_folder_out)
  return output_dirs


def _recursive_glob(treeroot, pattern):
  results = []
  for base, _, files in os.walk(treeroot):
    files = fnmatch.filter(files, pattern)
    results.extend(os.path.join(base, f) for f in files)
  return results

INFERENCE_MODE_SINGLE = 'single'  # Take plain single-frame input.
INFERENCE_MODE_TRIPLETS = 'triplets' 
INFERENCE_CROP_NONE = 'none'
INFERENCE_CROP_CITYSCAPES = 'cityscapes'
model_ckpt = 'model/KITTI/model-199160'

video_data = 'short_traffic_demo.mp4'
output_dir = 'test_output'

def _run_inference(output_dir=output_dir,
                   file_extension='png',
                   depth=True,
                   egomotion=False,
                   model_ckpt=model_ckpt,
                   input_list_file=None,
                   batch_size=1,
                   img_height=128,
                   img_width=416,
                   seq_length=3,
                   architecture=nets.RESNET,
                   imagenet_norm=True,
                   use_skip=True,
                   joint_encoder=True,
                   shuffle=False,
                   flip_for_depth=False,
                   inference_mode=INFERENCE_MODE_SINGLE,
                   inference_crop=INFERENCE_CROP_NONE,
                   use_masks=False):

  inference_model = model.Model(is_training=False,
                                batch_size=batch_size,
                                img_height=img_height,
                                img_width=img_width,
                                seq_length=seq_length,
                                architecture=architecture,
                                imagenet_norm=imagenet_norm,
                                use_skip=use_skip,
                                joint_encoder=joint_encoder)
  vars_to_restore = util.get_vars_to_save_and_restore(model_ckpt)
  saver = tf.train.Saver(vars_to_restore)
  sv = tf.train.Supervisor(logdir='/tmp/', saver=None)
  with sv.managed_session() as sess:
    saver.restore(sess, model_ckpt)
    if not gfile.Exists(output_dir):
      gfile.MakeDirs(output_dir)
    logging.info('Predictions will be saved in %s.', output_dir)

    #input camera image
    video_capture = cv2.VideoCapture(video_data)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_dir + '/' + 'try_1109.mp4', fourcc, fps, (416, 256))
    frame_count = (int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    while True:
      if depth:

        im_batch = []

        for i in range(frame_count):

          if i % 100 == 0:
            logging.info('%s of %s files processed.', i, range(frame_count))
          ret, im = video_capture.read()

          im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          #VGAの画像を切り出して416x128に合わせる
          #ymin, ymax, xmin, xmax = [142, 339, 0, 640]
          #im = im[ymin:ymax, xmin:xmax]
          im = cv2.resize(im, (img_width, img_height))
          im = np.array(im, dtype=np.float32) / 255.0

          im_batch.append(im)
          for _ in range(batch_size - len(im_batch)):  # Fill up batch.
            im_batch.append(np.zeros(shape=(img_height, img_width, 3), dtype=np.float32))

          im_batch = np.stack(im_batch, axis=0)
          est_depth = inference_model.inference_depth(im_batch, sess)
          
          color_map = util.normalize_depth_for_display(np.squeeze(est_depth))
          #color_map = (color_map * 255.0).astype(np.uint8)
          image_frame = np.concatenate((im_batch[0], color_map), axis=0)
          image_frame = (image_frame * 255.0).astype(np.uint8)
          image_frame = cv2.cvtColor(image_frame, cv2.COLOR_RGB2BGR)
          
          #YOLO here
          with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
              for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(image_frame)
                strip_optimizer(opt.weights)
            else:
              detect(image_frame)


          ##inout yolo detector around here!

          out.write(image_frame)
          im_batch = []

    logging.info('Done.')
    video_capture.release()
    out.release()


def main(_):
  _run_inference()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    app.run(main)