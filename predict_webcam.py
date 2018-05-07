import multiprocessing
import time
from multiprocessing import set_start_method

import cv2

from models.model_openpose import OpenPoseModelHandler

try:
    set_start_method('spawn')
except RuntimeError:
    pass
from utils import util
from skeletons.skeleton_config_rtpose2d import SkeletonConfigRtPose2D
from network import get_network_results

from utils.util_predict import get_human_data
from visualization import get_human_pose_image

fps_time = 0


@util.measure_time
def predict(model, img_result, skeleton_config, input_img):
    global fps_time
    start_time = time.time()
    joint_positions, limbs, humans = get_human_data(model, img_result, input_img, skeleton_config)
    print("{}: {}".format("get poses", time.time() - start_time))
    pose_img = get_human_pose_image(input_img, joint_positions, humans, SkeletonConfigRtPose2D.limbs, SkeletonConfigRtPose2D.limb_colors)
    cv2.putText(pose_img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.imshow('webcam', pose_img)
    if cv2.waitKey(1) == 27:
        exit(0)
    fps_time = time.time()


def cam_loop(pipe_parent):
    cap = cv2.VideoCapture(0)
    model = OpenPoseModelHandler().get_model()
    OpenPoseModelHandler().load_state_dict(model)
    model.eval()
    while True:
        ret_val, img = cap.read()
        if img is not None:
            img_result = get_network_results(model, img)
            pipe_parent.send((model, img, img_result))


def show_loop(pipe_child):
    cv2.namedWindow('webcam')

    skeleton_config = SkeletonConfigRtPose2D()

    while True:
        from_queue = pipe_child.recv()
        predict(from_queue[0], from_queue[2], skeleton_config, from_queue[1])


if __name__ == "__main__":
    pipe_parent, pipe_child = multiprocessing.Pipe()

    cam_process = multiprocessing.Process(target=cam_loop, args=(pipe_parent, ))
    cam_process.start()

    show_process = multiprocessing.Process(target=show_loop, args=(pipe_child, ))
    show_process.start()

    cam_process.join()
    show_process.join()

