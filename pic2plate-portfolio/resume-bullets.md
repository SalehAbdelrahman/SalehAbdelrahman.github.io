**Pic2Plate: Open-Vocabulary Food Perception** | Python, PyTorch, Transformers, OWLv2, Faster R-CNN, Claude API, Flask

- Built a two-stage perception pipeline that combines a vision-language model (structured outputs) for
  item identification with OWLv2 open-vocabulary detection for localization, running image encoding
  in parallel with the VLM call to hide its latency
- Raised multi-item recall from 2% to 80% (F1 0.04 → 0.80) with a detect-then-classify pipeline: COCO
  Faster R-CNN plus class-agnostic RPN proposals, a ResNet-18 (81 classes) fine-tuned with crop
  augmentation on each crop (+0.11 F1), and confidence voting tuned on the validation split
- Designed a reproducible benchmark of randomized multi-item composite scenes with cached detections
  for fast threshold sweeps; used failure analysis on real photos to justify moving from a closed to an
  open vocabulary
- Shipped a Flask web app with camera capture, a bounding-box viewer, and pantry-aware recipe ranking

Shorter, 2-bullet version:

- Built a two-stage food perception pipeline: VLM structured extraction for identification and OWLv2
  open-vocabulary detection for localization, with image encoding run in parallel with the VLM call
- Benchmarked a detect-then-classify baseline (Faster R-CNN + RPN proposals → fine-tuned ResNet-18)
  on randomized multi-item scenes, raising recall from 2% to 80% (F1 0.80), then used failure
  analysis on real photos to justify the move to open-vocabulary detection
