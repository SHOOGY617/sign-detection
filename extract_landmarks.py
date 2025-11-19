# extract_landmarks.py
import os
# Reduce TensorFlow/MediaPipe INFO/WARNING spam when importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_from_image(img_path, hands, to_pixels=False):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    h, w, _ = img.shape
    coords = []
    for p in lm.landmark:
        coords.append(p.x)  # normalized
        coords.append(p.y)
    if to_pixels:
        pix = []
        for i in range(0, len(coords), 2):
            x_n = coords[i]
            y_n = coords[i+1]
            pix.append(x_n * w)
            pix.append(y_n * h)
        return pix
    return coords  # length 42 (21*2)

def main(dataset_dir, out_csv="landmarks.csv", to_pixels=False):
    rows = []
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print("Found classes:", classes)
    # create MediaPipe Hands inside a context manager so it is automatically closed
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        for cls in classes:
            cls_dir = os.path.join(dataset_dir, cls)
            for fname in tqdm(os.listdir(cls_dir), desc=f"Processing {cls}"):
                if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
                    continue
                path = os.path.join(cls_dir, fname)
                coords = extract_from_image(path, hands, to_pixels=to_pixels)
                if coords is None:
                    continue
                rows.append([cls] + coords)

    if not rows:
        print("No landmarks found. Try increasing detection confidence or use clearer images.")
        return

    cols = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
    # we built coords as x0,y0,x1,y1... -> need to convert to x0..x20,y0..y20
    df_rows = []
    for r in rows:
        label = r[0]
        coords = r[1:]
        xs = coords[0::2]
        ys = coords[1::2]
        df_rows.append([label] + xs + ys)
    df = pd.DataFrame(df_rows, columns=cols)
    df.to_csv(out_csv, index=False)
    print("Saved landmarks CSV to", out_csv)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset dir (folders of classes)")
    parser.add_argument("--out", default="landmarks.csv", help="Output CSV path")
    parser.add_argument("--pixels", action="store_true", help="Convert normalized landmarks to pixel coordinates")
    args = parser.parse_args()
    main(args.data, args.out, to_pixels=args.pixels)
