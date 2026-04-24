import sys
sys.path.insert(0, '/Users/artemijmosin/Documents/projects/slovo')
import onnxruntime as ort
import numpy as np
import cv2
from constants import classes
import os

VIDEOS = [
    'examples/0a2ffece-2832-4011-b656-915f39aa7850.mp4',
    'examples/2f39d6e2-695f-4238-8061-8764b998de5d.mp4',
    'examples/9a97a1f2-7404-4d8a-bc57-dfa87fa42e93.mp4',
    'examples/f17a6060-6ced-4bd1-9886-8578cfbb864f.mp4',
]

mean = np.array([123.675, 116.28, 103.53])
std  = np.array([58.395, 57.12, 57.375])


def resize(im, new_shape=(224, 224)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im,
                            int(round(dh - 0.1)), int(round(dh + 0.1)),
                            int(round(dw - 0.1)), int(round(dw + 0.1)),
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im


sessions = {}
for model_name, frame_interval in [('mvit32-2.onnx', 2), ('SignFlow-R.onnx', 1)]:
    sessions[model_name] = (ort.InferenceSession(f'/Users/artemijmosin/Documents/projects/slovo/{model_name}'), frame_interval)

for video_path in VIDEOS:
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f'\n=== {os.path.basename(video_path)[:8]}…  ({len(raw_frames)} кадров) ===')

    for model_name, (session, frame_interval) in sessions.items():
        input_name  = session.get_inputs()[0].name
        window_size = session.get_inputs()[0].shape[3]
        out_names   = [o.name for o in session.get_outputs()]

        tensors_list    = []
        prediction_list = ['---']
        frame_counter   = 0

        for frame_idx, frame in enumerate(raw_frames):
            frame_counter += 1
            if frame_counter == frame_interval:
                img = resize(frame.astype(np.float32))
                img = (img - mean) / std
                img = np.transpose(img, [2, 0, 1])
                tensors_list.append(img)
                frame_counter = 0

            if len(tensors_list) >= window_size:
                inp  = np.stack(tensors_list[:window_size], axis=1)[None][None].astype(np.float32)
                out  = session.run(out_names, {input_name: inp})[0].squeeze()
                gloss = classes.get(int(out.argmax()), f'cls_{out.argmax()}')
                conf  = float(out.max())

                if gloss != prediction_list[-1] and gloss != '---':
                    prediction_list.append(gloss)

                for _ in range(window_size):
                    if tensors_list:
                        tensors_list.pop(0)

        print(f'  [{model_name}] -> {" | ".join(prediction_list[1:]) or "(нет предсказаний)"}')
