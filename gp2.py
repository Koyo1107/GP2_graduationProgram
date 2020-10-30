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
import torch
import argparse
from utils.datasets import *
from utils.utils import *
from models import *
#------------------

INFERENCE_MODE_SINGLE = 'single'  # Take plain single-frame input.
INFERENCE_MODE_TRIPLETS = 'triplets'  # Take image triplets as input.
# For KITTI, we just resize input images and do not perform cropping. For
# Cityscapes, the car hood and more image content has been cropped in order
# to fit aspect ratio, and remove static content from the images. This has to be
# kept at inference time.
INFERENCE_CROP_NONE = 'none'
INFERENCE_CROP_CITYSCAPES = 'cityscapes'

input_dir = 'input'
output_dir = 'semioutput'
model_ckpt = 'model/KITTI/model-199160'

def _run_inference(output_dir=output_dir,
                   file_extension='png',
                   depth=True,
                   egomotion=True,
                   model_ckpt=model_ckpt,
                   input_dir=input_dir,
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

    # Collect all images to run inference on.
    im_files, basepath_in = collect_input_images(input_dir, input_list_file,
                                                 file_extension)
    if shuffle:
      logging.info('Shuffling data...')
      np.random.shuffle(im_files)
    logging.info('Running inference on %d files.', len(im_files))

    # Create missing output folders and pre-compute target directories.
    output_dirs = create_output_dirs(im_files, basepath_in, output_dir)

    # Run depth prediction network.
    if depth:
      im_batch = []
      for i in range(len(im_files)):
        if i % 100 == 0:
          logging.info('%s of %s files processed.', i, len(im_files))

        # Read image and run inference.
        if inference_mode == INFERENCE_MODE_SINGLE:
          if inference_crop == INFERENCE_CROP_NONE:
            im = util.load_image(im_files[i], resize=(img_width, img_height))
          elif inference_crop == INFERENCE_CROP_CITYSCAPES:
            im = util.crop_cityscapes(util.load_image(im_files[i]),
                                      resize=(img_width, img_height))
        elif inference_mode == INFERENCE_MODE_TRIPLETS:
          im = util.load_image(im_files[i], resize=(img_width * 3, img_height))
          im = im[:, img_width:img_width*2]
        if flip_for_depth:
          im = np.flip(im, axis=1)
        im_batch.append(im)

        if len(im_batch) == batch_size or i == len(im_files) - 1:
          # Call inference on batch.
          for _ in range(batch_size - len(im_batch)):  # Fill up batch.
            im_batch.append(np.zeros(shape=(img_height, img_width, 3),
                                     dtype=np.float32))
          im_batch = np.stack(im_batch, axis=0)
          est_depth = inference_model.inference_depth(im_batch, sess)
          if flip_for_depth:
            est_depth = np.flip(est_depth, axis=2)
            im_batch = np.flip(im_batch, axis=2)

          for j in range(len(im_batch)):
            color_map = util.normalize_depth_for_display(np.squeeze(est_depth[j]))
            visualization = np.concatenate((im_batch[j], color_map), axis=0)
            # Save raw prediction and color visualization. Extract filename
            # without extension from full path: e.g. path/to/input_dir/folder1/
            # file1.png -> file1
            k = i - len(im_batch) + 1 + j
            filename_root = os.path.splitext(os.path.basename(im_files[k]))[0]
            pref = '_flip' if flip_for_depth else ''
            #output_raw = os.path.join(output_dirs[k], filename_root + pref + '.npy')
            output_vis = os.path.join(output_dirs[k], filename_root + pref + '.png')
            #with gfile.Open(output_raw, 'wb') as f:
            #  np.save(f, est_depth[j])
            util.save_image(output_vis, visualization, file_extension)
          #inputsource = im_batch
          #print ('Print here im_batch test')
          #print (len(im_batch))
          im_batch = []

    # Run egomotion network.
    if egomotion:
      if inference_mode == INFERENCE_MODE_SINGLE:
        # Run regular egomotion inference loop.
        input_image_seq = []
        input_seg_seq = []
        current_sequence_dir = None
        current_output_handle = None
        for i in range(len(im_files)):
          sequence_dir = os.path.dirname(im_files[i])
          if sequence_dir != current_sequence_dir:
            # Assume start of a new sequence, since this image lies in a
            # different directory than the previous ones.
            # Clear egomotion input buffer.
            output_filepath = os.path.join(output_dirs[i], 'egomotion.txt')
            if current_output_handle is not None:
              current_output_handle.close()
            current_sequence_dir = sequence_dir
            logging.info('Writing egomotion sequence to %s.', output_filepath)
            current_output_handle = gfile.Open(output_filepath, 'w')
            input_image_seq = []
          im = util.load_image(im_files[i], resize=(img_width, img_height))
          input_image_seq.append(im)
          if use_masks:
            im_seg_path = im_files[i].replace('.%s' % file_extension,
                                              '-seg.%s' % file_extension)
            if not gfile.Exists(im_seg_path):
              raise ValueError('No segmentation mask %s has been found for '
                               'image %s. If none are available, disable '
                               'use_masks.' % (im_seg_path, im_files[i]))
            input_seg_seq.append(util.load_image(im_seg_path,
                                                 resize=(img_width, img_height),
                                                 interpolation='nn'))

          if len(input_image_seq) < seq_length:  # Buffer not filled yet.
            continue
          if len(input_image_seq) > seq_length:  # Remove oldest entry.
            del input_image_seq[0]
            if use_masks:
              del input_seg_seq[0]

          input_image_stack = np.concatenate(input_image_seq, axis=2)
          input_image_stack = np.expand_dims(input_image_stack, axis=0)
          if use_masks:
            input_image_stack = mask_image_stack(input_image_stack,
                                                 input_seg_seq)
          est_egomotion = np.squeeze(inference_model.inference_egomotion(
              input_image_stack, sess))
          egomotion_str = []
          for j in range(seq_length - 1):
            egomotion_str.append(','.join([str(d) for d in est_egomotion[j]]))
          current_output_handle.write(
              str(i) + ' ' + ' '.join(egomotion_str) + '\n')
        if current_output_handle is not None:
          current_output_handle.close()
      elif inference_mode == INFERENCE_MODE_TRIPLETS:
        written_before = []
        for i in range(len(im_files)):
          im = util.load_image(im_files[i], resize=(img_width * 3, img_height))
          input_image_stack = np.concatenate(
              [im[:, :img_width], im[:, img_width:img_width*2],
               im[:, img_width*2:]], axis=2)
          input_image_stack = np.expand_dims(input_image_stack, axis=0)
          if use_masks:
            im_seg_path = im_files[i].replace('.%s' % file_extension,
                                              '-seg.%s' % file_extension)
            if not gfile.Exists(im_seg_path):
              raise ValueError('No segmentation mask %s has been found for '
                               'image %s. If none are available, disable '
                               'use_masks.' % (im_seg_path, im_files[i]))
            seg = util.load_image(im_seg_path,
                                  resize=(img_width * 3, img_height),
                                  interpolation='nn')
            input_seg_seq = [seg[:, :img_width], seg[:, img_width:img_width*2],
                             seg[:, img_width*2:]]
            input_image_stack = mask_image_stack(input_image_stack,
                                                 input_seg_seq)
          est_egomotion = inference_model.inference_egomotion(
              input_image_stack, sess)
          est_egomotion = np.squeeze(est_egomotion)
          egomotion_1_2 = ','.join([str(d) for d in est_egomotion[0]])
          egomotion_2_3 = ','.join([str(d) for d in est_egomotion[1]])

          output_filepath = os.path.join(output_dirs[i], 'egomotion.txt')
          file_mode = 'w' if output_filepath not in written_before else 'a'
          with gfile.Open(output_filepath, file_mode) as current_output_handle:
            current_output_handle.write(str(i) + ' ' + egomotion_1_2 + ' ' +
                                        egomotion_2_3 + '\n')
          written_before.append(output_filepath)
      logging.info('Done.')

def mask_image_stack(input_image_stack, input_seg_seq):
  """Masks out moving image contents by using the segmentation masks provided.
  This can lead to better odometry accuracy for motion models, but is optional
  to use. Is only called if use_masks is enabled.
  Args:
    input_image_stack: The input image stack of shape (1, H, W, seq_length).
    input_seg_seq: List of segmentation masks with seq_length elements of shape
                   (H, W, C) for some number of channels C.
  Returns:
    Input image stack with detections provided by segmentation mask removed.
  """
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


def collect_input_images(input_dir, input_list_file, file_extension):
  """Collects all input images that are to be processed."""
  if input_dir is not None:
    im_files = _recursive_glob(input_dir, '*.' + file_extension)
    basepath_in = os.path.normpath(input_dir)
  elif input_list_file is not None:
    im_files = util.read_text_lines(input_list_file)
    basepath_in = os.path.dirname(input_list_file)
    im_files = [os.path.join(basepath_in, f) for f in im_files]
  im_files = [f for f in im_files if 'disp' not in f and '-seg' not in f and
              '-fseg' not in f and '-flip' not in f]
  return sorted(im_files), basepath_in


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

#----------------------------------------------------------------------------------------------------------------------
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
    print('wote on the depth')

#yolov3
def detect(save_img=False): 
    img_size = 512 
    out = 'output'
    source = 'semioutput'
    weights = 'weights/yolov3-spp-ultralytics.pt'
    cfg_ = 'cfg/yolov3-spp.cfg'
    names_ = 'data/coco.names'
    fourcc = 'mp4'
    half = opt.half
    view_img = opt.view_img
    save_txt = opt.save_txt

    imgsz = img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(cfg_, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(names_)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_bbox_and_depth(xyxy, im0, label=label, color=colors[int(cls)])
                        color_detection(im0, xyxy, color=colors[int(cls)])


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

def main(_):
  _run_inference()
  detect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()

    app.run(main)
