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

INFERENCE_MODE_SINGLE = 'single'  # Take plain single-frame input.
INFERENCE_MODE_TRIPLETS = 'triplets'  # Take image triplets as input.
# For KITTI, we just resize input images and do not perform cropping. For
# Cityscapes, the car hood and more image content has been cropped in order
# to fit aspect ratio, and remove static content from the images. This has to be
# kept at inference time.
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
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_dir + '/' + 'try_1106.mp4', fourcc, fps, (img_width, img_height))

    while True:
        if depth:
            im_batch = []
            ret, im = video_capture.read()

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #VGAの画像を切り出して416x128に合わせる
            ymin, ymax, xmin, xmax = [142, 339, 0, 640]
            im = im[ymin:ymax, xmin:xmax]
            im = cv2.resize(im, (img_width,img_height))
            im_another = im
            im = np.array(im, dtype=np.float32) / 255.0
            im_batch.append(im)
            im_another = cv2.cvtColor(im_another, cv2.COLOR_RGB2BGR)

            for _ in range(batch_size - len(im_batch)):  # Fill up batch.
                im_batch.append(np.zeros(shape=(img_height, img_width, 3),dtype=np.float32))
            im_batch = np.stack(im_batch, axis=0)
            est_depth = inference_model.inference_depth(im_batch, sess)

            color_map = util.normalize_depth_for_display(np.squeeze(est_depth))
            image_frame = np.concatenate((im_another, color_map), axis=0)
            image_frame = (image_frame * 255.0).astype(np.uint8)
            image_frame = cv2.cvtColor(image_frame, cv2.COLOR_RGB2BGR)

            out.write(image_frame)
            #logging.info('Frame Written')
            im_batch = []
    logging.info('Done.')
    video_capture.release()
    out.release()

    
def main(_):
  _run_inference()
  #detect()

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    #parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    #parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    #parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    #parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    #parser.add_argument('--augment', action='store_true', help='augmented inference')
    #parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    #parser.add_argument('--view-img', action='store_true', help='display results')
    #parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    #opt = parser.parse_args()
    app.run(main)