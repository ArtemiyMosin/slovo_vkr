import sys
sys.path.insert(0, '/Users/artemijmosin/Documents/projects/slovo')
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from constants import classes

# Load GIF frames
gif = Image.open('/Users/artemijmosin/Downloads/demo.gif')
raw_frames = []
try:
    while True:
        raw_frames.append(np.array(gif.convert('RGB')))
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

# Crop bottom 50px text strip (baked in by demo.py renderer)
TEXT_STRIP = 50
raw_frames = [f[:-TEXT_STRIP] for f in raw_frames]
print(f'GIF: {len(raw_frames)} кадров, размер после обрезки: {raw_frames[0].shape[:2]}')


def resize(im, new_shape=(224, 224)):
    """Exact copy of Runner.resize() from demo.py."""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im


mean = np.array([123.675, 116.28, 103.53])
std  = np.array([58.395, 57.12, 57.375])

for model_name, frame_interval in [('mvit32-2.onnx', 2), ('SignFlow-R.onnx', 1)]:
    session = ort.InferenceSession(f'/Users/artemijmosin/Documents/projects/slovo/{model_name}')
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
                print(f'  [{model_name}] кадр ~{frame_idx:3d}  ->  "{gloss}"  conf={conf*100:.1f}%')

            for _ in range(window_size):
                if tensors_list:
                    tensors_list.pop(0)

    print(f'\n[{model_name}] Последовательность: {" | ".join(prediction_list[1:])}')
    print()
