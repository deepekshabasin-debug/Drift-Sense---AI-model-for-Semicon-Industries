# Drift-Sense

## AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

Drift-Sense is an AI-powered computer vision solution designed to recover
navigation errors in wafer inspection tools.

## Problem

Wafer inspection tools must repeatedly return to the same site on a die.
Thermal expansion, vibration and mechanical errors can cause navigation
drift.

Highly periodic semiconductor layouts make conventional template matching
difficult because many regions appear visually similar.

## Our Solution

Our system uses computer vision and deep learning to identify the target
location in a Search Image using a Reference Image.

### Pipeline

Reference Image
        ↓
Image Preprocessing
        ↓
Feature / Candidate Processing
        ↓
Deep Learning Model
        ↓
Target Localization
        ↓
Predicted (X,Y)

## Project Files

- `app.py` — Main application
- `requirements.txt` — Required Python libraries
- `drift_sense_rgb_model.pth` — Trained PyTorch model

## Technologies

- Python
- OpenCV
- PyTorch
- NumPy
- Computer Vision
- Deep Learning
- ## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt 
```
## Prototype Demonstration

[▶️ Watch the Drift-Sense Prototype Demo](https://youtu.be/Osh65-iA9QY?si=DT2kmO6Ixzvd4u2B)
