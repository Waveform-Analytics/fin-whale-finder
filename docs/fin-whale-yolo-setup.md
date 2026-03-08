# Fin Whale YOLO Setup

## 1. Create Repo

```bash
# Create repo on GitHub called "fin-whale-yolo"
# Clone locally:
git clone https://github.com/YOUR_USERNAME/fin-whale-yolo.git
cd fin-whale-yolo
```

## 2. Initialize Speckit

```bash
uvx speckit init
# Follow prompts - name: fin-whale-yolo, type: package
```

## 3. Add Dependencies

Edit `pyproject.toml` to add:

```toml
[project]
dependencies = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "torch>=2.0",
    "torchvision>=0.15",
    "ultralytics>=8.0",
    "opencv-python",
]
```

Then:
```bash
uv sync
```

## 4. Get Data

### Option A: Copy from fin-whale-finder
```bash
# From this repo's data/cache/ directory
cp -r ../fin-whale-finder/data/cache .
```

### Option B: Re-fetch from OOI
```bash
# Make sure ~/.netrc has OOI credentials
# Edit scripts/fetch_week_data.py to point to your new repo
uv run python scripts/fetch_week_data.py
```

## 5. Generate Spectrograms

Create `scripts/generate_spectrograms.py`:

```python
"""Generate short spectrogram clips for YOLO labeling."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy.signal import spectrogram
import argparse

# Params for fin whale band (15-30 Hz)
FFT_WINDOW = 5  # seconds
OVERLAP = 0.5   # 50% overlap
FREQ_MIN = 10
FREQ_MAX = 50

def load_audio(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['audio'], data['sample_rate']

def make_spectrogram(audio, sr, fft_win=FFT_WINDOW, overlap=OVERLAP):
    nperseg = int(fft_win * sr)
    noverlap = int(nperseg * overlap)
    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    
    # Frequency mask
    fmask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
    Sxx = Sxx[fmask, :]
    f = f[fmask]
    
    return t, f, Sxx

def save_spectrogram_image(Sxx, t, f, output_path, duration_sec):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(10 * np.log10(Sxx + 1e-10), aspect='auto', origin='lower',
              extent=[t[0], t[-1], f[0], f[-1]], cmap='magma')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(FREQ_MIN, FREQ_MAX)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, default=Path('data/cache'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/spectrograms'))
    parser.add_argument('--clip-length', type=int, default=120, help='seconds')
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_files = sorted(args.cache_dir.glob('*.pkl'))
    print(f"Found {len(cache_files)} cache files")
    
    for pkl_file in cache_files:
        print(f"Processing {pkl_file.name}...")
        audio, sr = load_audio(pkl_file)
        
        # Split into short clips
        samples_per_clip = args.clip_length * sr
        n_clips = len(audio) // samples_per_clip
        
        for i in range(n_clips):
            start = i * samples_per_clip
            end = start + samples_per_clip
            clip_audio = audio[start:end]
            
            t, f, Sxx = make_spectrogram(clip_audio, sr)
            
            output_name = f"{pkl_file.stem}_clip{i:03d}.png"
            output_path = args.output_dir / output_name
            
            save_spectrogram_image(Sxx, t, f, output_path, args.clip_length)
    
    print(f"Saved spectrograms to {args.output_dir}")

if __name__ == '__main__':
    main()
```

Run:
```bash
uv run python scripts/generate_spectrograms.py --clip-length 120
```

## 6. Labeling Tool

Create `scripts/label_tool.py`:

```python
"""Simple bbox labeling tool for spectrograms."""
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

class SimpleLabeler:
    def __init__(self, image_dir, output_file):
        self.image_dir = Path(image_dir)
        self.output_file = Path(output_file)
        self.labels = {}
        if self.output_file.exists():
            with open(self.output_file) as f:
                self.labels = json.load(f)
        
        self.images = sorted([f for f in self.image_dir.glob('*.png')])
        self.current_idx = 0
        self.boxes = []
        self.drawing = False
        self.start_xy = None
        
        print(f"Loaded {len(self.images)} images")
        print("Controls: [d] next, [a] prev, [s] save, [c] clear, [q] quit")
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_xy = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img = self.current_image.copy()
            cv2.rectangle(img, self.start_xy, (x, y), (0, 255, 0), 2)
            cv2.imshow('Labeler', img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self.start_xy
            x2, y2 = x, y
            # Normalize to 0-1
            h, w = self.current_image.shape[:2]
            self.boxes.append([x1/w, y1/h, x2/w, y2/h])
            print(f"Box added: {len(self.boxes)} total")
    
    def run(self):
        while 0 <= self.current_idx < len(self.images):
            img_path = self.images[self.current_idx]
            img_name = img_path.name
            
            self.current_image = cv2.imread(str(img_path))
            self.boxes = self.labels.get(img_name, {}).get('boxes', [])
            
            print(f"\n{img_name} ({self.current_idx+1}/{len(self.images)})")
            print(f"Boxes: {len(self.boxes)}")
            
            cv2.namedWindow('Labeler')
            cv2.setMouseCallback('Labeler', self.mouse_callback)
            cv2.imshow('Labeler', self.current_image)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.save_current()
                self.current_idx += 1
            elif key == ord('a'):
                self.save_current()
                self.current_idx -= 1
            elif key == ord('s'):
                self.save_current()
            elif key == ord('c'):
                self.boxes = []
            
            cv2.destroyAllWindows()
        
        self.save_current()
        print("Done!")
    
    def save_current(self):
        img_name = self.images[self.current_idx].name
        self.labels[img_name] = {'boxes': self.boxes, 'label': 'call'}
        with open(self.output_file, 'w') as f:
            json.dump(self.labels, f, indent=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=Path, default=Path('data/spectrograms'))
    parser.add_argument('--output', type=Path, default=Path('data/labels.json'))
    args = parser.parse_args()
    
    labeler = SimpleLabeler(args.image_dir, args.output)
    labeler.run()
```

Run:
```bash
uv run python scripts/label_tool.py
```

Controls:
- **Mouse drag** — draw box
- **d** — next image
- **a** — previous image
- **s** — save
- **c** — clear boxes
- **q** — quit

## 7. Convert Labels to YOLO Format

Create `scripts/convert_to_yolo.py`:

```python
"""Convert simple JSON labels to YOLO format."""
import json
import yaml
from pathlib import Path

def convert_to_yolo(label_file, image_dir, output_dir):
    with open(label_file) as f:
        labels = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLO needs class index (0 for 'call')
    for img_name, data in labels.items():
        img_path = Path(image_dir) / img_name
        if not img_path.exists():
            continue
        
        import cv2
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        txt_name = img_name.replace('.png', '.txt')
        txt_path = output_dir / txt_name
        
        with open(txt_path, 'w') as f:
            for box in data['boxes']:
                x1, y1, x2, y2 = box
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                f.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")
    
    print(f"Wrote YOLO labels to {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=Path, default=Path('data/labels.json'))
    parser.add_argument('--image-dir', type=Path, default=Path('data/spectrograms'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/yolo_labels'))
    args = parser.parse_args()
    
    convert_to_yolo(args.label_file, args.image_dir, args.output_dir)
```

## 8. Create YOLO Dataset Config

Create `data/yolo_dataset.yaml`:

```yaml
path: data
train: spectrograms
val: spectrograms

names:
  0: call
```

Note: For now, put some images in both train and val folders, or create a proper split later.

## 9. Train YOLO

```bash
uv run python -c "
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano model - fast, good for starting
results = model.train(
    data='data/yolo_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    project='runs/detect',
    name='fin_whale'
)
"
```

Or create a training script `scripts/train.py`.

## 10. Inference

Create `scripts/detect.py`:

```python
"""Run detection on new spectrograms."""
from ultralytics import YOLO
from pathlib import Path

def detect(model_path, image_dir, output_dir):
    model = YOLO(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = model(image_dir, save=True, project=output_dir, name='predictions')
    
    # Extract call timestamps
    for r in results:
        boxes = r.boxes
        print(f'{Path(r.path).name}: {len(boxes)} calls')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default='runs/detect/fin_whale/weights/best.pt')
    parser.add_argument('--image-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/detect/predictions'))
    args = parser.parse_args()
    
    detect(args.model, args.image_dir, args.output_dir)
```

Run:
```bash
uv run python scripts/detect.py --image-dir data/spectrograms
```

## Tips

1. Start with 50-100 labeled boxes — enough to test the pipeline
2. Use YOLOv8n (nano) for fast training on laptop
3. Data augmentation: YOLO does this automatically
4. If you need more labels, use active learning — let model predict on unlabeled, review uncertain ones first

## Next Steps After Getting Started

- Expand to more data
- Try YOLOv8s or YOLOv8m if nano underfits
- Convert bbox coordinates back to timestamps for IPI analysis
- Compare with your 2017 matched-filter results
