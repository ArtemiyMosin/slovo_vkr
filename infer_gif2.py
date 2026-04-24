import sys
sys.path.insert(0, '/Users/artemijmosin/Documents/projects/slovo')
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from constants import classes

gif = Image.open('/Users/artemijmosin/Downloads/demo.gif')
raw_frames = []
try:
    while True:
        raw_frames.append(np.array(gif.convert('RGB')))
        gif.seek(gif.tell() + 1)
except EOFError:
    pass
print(f'GIF: {len(raw_frames)} кадров')

def resize_pad(im, s=224):
    h, w = im.shape[:2]
    r = min(s/h, s/w)
    nh, nw = int(h*r), int(w*r)
    im = cv2.resize(im, (nw, nh))
    ph, pw = s-nh, s-nw
    return np.pad(im, ((ph//2, ph-ph//2), (pw//2, pw-pw//2), (0,0)), constant_values=114)

mean = np.array([123.675, 116.28, 103.53])
std  = np.array([58.395, 57.12, 57.375])

for model_name, frame_interval in [('SignFlow-R.onnx', 1), ('mvit32-2.onnx', 2)]:
    session = ort.InferenceSession(f'/Users/artemijmosin/Documents/projects/slovo/{model_name}')
    input_name = session.get_inputs()[0].name
    window_size = session.get_inputs()[0].shape[3]
    out_names = [o.name for o in session.get_outputs()]

    tensors_list = []
    prediction_list = ['---']
    frame_counter = 0

    for frame_idx, frame in enumerate(raw_frames):
        frame_counter += 1
        if frame_counter == frame_interval:
            img = resize_pad(frame.astype(np.float32))
            img = (img - mean) / std
            img = np.transpose(img, [2, 0, 1])
            tensors_list.append(img)
            frame_counter = 0

        if len(tensors_list) >= window_size:
            inp = np.stack(tensors_list[:window_size], axis=1)[None][None].astype(np.float32)
            out = session.run(out_names, {input_name: inp})[0].squeeze()
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
