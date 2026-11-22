# WildfireGuard AI

WildfireGuard AI is a decision-support prototype that detects visible fire and
smoke in images, sampled video frames, and live webcam or RTSP/RTSPS streams. It
combines the project's YOLO labels with its Pascal VOC XML annotations and
trains a two-class YOLO detector.

## Setup

From this project folder, install the dependencies and train the detector:

```powershell
pip install -r requirements.txt
python train.py --epochs 10 --image-size 416 --batch 4 --device cpu
python -m src.prediction.prepare_spread_data
python -m src.prediction.build_spread_targets
python -m src.prediction.train_area_forecast_model
streamlit run app.py
```

The training step prepares `data/processed/` and writes the best model to
`models/wildfire_yolo_v2.pt`, which the app loads. Ultralytics downloads its
small pretrained starting weights the first time training runs, so that step
needs an internet connection. Use `--device 0` when a compatible CUDA GPU is
available. CPU training can be slow. The project data is limited in real-world
diversity. Smoke-only scenes may therefore be missed; use diverse,
accurately labeled real footage and evaluate per-class precision and recall
before relying on a model in the field.

## Optional synthetic fire/smoke training data

The training pipeline also accepts the [Fire and Smoke Synthetic 10k
dataset](https://huggingface.co/datasets/athkot/fire-and-smoke-synthetic-10k)
when its YOLO export is extracted under
`data/external/fire-and-smoke-synthetic-10k/`. It uses the expected class IDs
`0=fire` and `1=smoke`. The dataset is licensed CC BY 4.0; credit aionVIS when
using it. It contains AI-generated images, not real wildfire photographs, and
has not been benchmarked by its authors on real photos, so improvements on
real-world images are not guaranteed. Training uses the dataset's training
split and leaves its original validation split untouched.
One fine-tuning trial improved synthetic validation but reduced smoke recall
on the project's real-image holdout, so its checkpoint was rejected. Validate
any retrained checkpoint on real imagery before replacing the active model.

To download and extract the roughly 910 MB YOLO export from PowerShell:

```powershell
New-Item -ItemType Directory -Force data\external\fire-and-smoke-synthetic-10k
curl.exe -L --fail -o data\external\fire-and-smoke-synthetic-10k.zip "https://huggingface.co/datasets/athkot/fire-and-smoke-synthetic-10k/resolve/main/yolo/dataset-yolo.zip?download=true"
Expand-Archive -LiteralPath data\external\fire-and-smoke-synthetic-10k.zip -DestinationPath data\external\fire-and-smoke-synthetic-10k -Force
```

The spread-model commands combine the included Canadian fire-growth records,
create next-day targets, and train `models/wildfire_area_forecast.pkl`.

## What the outputs mean

- **Fire/smoke detections** are object boxes and confidence scores from the
  trained YOLO model. The displayed risk label is a rule-based visual triage cue,
  not an operational fire-danger rating. Recommendations are advisory only.
  Lower confidence thresholds can reveal weaker candidates but may greatly
  increase false alarms.
- **Visible area** is a bounding-box image fraction. Hectares are only estimated
  when the operator supplies the ground area visible in that view; the boxes
  include background and are not a burned-area segmentation. The resulting
  camera fire-box hectares are a rough visual estimate, not verified real-world
  fire area.
- **Spread estimates** are generated after image, video, and live-camera analysis
  when the operator supplies coordinates and the ground area visible in the
  camera view. The app fetches weather from Open-Meteo (only coordinates are
  sent), converts calibrated fire-box coverage into a rough current area proxy,
  and predicts the next recorded day's `firearea` value using a location-based
  regression model trained on Canadian fire-growth records. This target is not a
  cumulative total burned area. The current camera-derived proxy and model target
  are not directly like-for-like. The held-out fire-level test MAE was about
  425 ha and R² was 0.066. This experimental model is not validated outside its
  Canadian training region, and its output is not reliable for operational
  decisions. It predicts numeric next-day daily fire-area growth, not a geographic
  spread path, fire perimeter, or direction of spread.
- **Separate spread classifier:** the SGD classifier experiment uses 42 tabular
  features to classify significant next-day growth. It is not integrated into
  the Streamlit prediction workflow; the application does not retrieve or fill
  its terrain, fire-history, or other tabular inputs.
- **Video analysis** processes sampled frames in bounded batches to reduce
  inference overhead. A shorter sampling interval examines more frames but takes
  longer. Fire and smoke observation times are reported separately and are not
  confirmed ignition or extinguishment times. A CSV report can be downloaded
  with timestamps, detections, confidences, and box coordinates.
- **Live monitoring** reads from a camera connected to the computer running the
  app or from an RTSP/RTSPS CCTV stream. Frames are analyzed in memory and are
  not recorded. The operator must stop monitoring to release the camera.

This software can miss fires, confuse smoke with clouds or fog, and produce
false alarms. Risk recommendations are advisory; the software does not provide
autonomous evacuation or firefighting instructions. Do not use it as the sole
basis for an evacuation, dispatch, or fire-suppression decision. Follow local
emergency guidance and contact emergency services when needed.
