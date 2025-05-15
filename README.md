
# IoV Auto-Authentication System

An adaptive identity authentication framework for the Internet of Vehicles (IoV), dynamically selecting authentication strategies based on real-time environmental sensing and risk assessment. This project combines multimodal perception with Z3-based fuzzy reasoning to optimize the trade-off between security and latency in intelligent transportation systems.

## 📝 Project Overview

With the rapid growth of IoV, secure and efficient identity authentication has become critical. Traditional static methods struggle with dynamic, high-concurrency environments. This system addresses those challenges by:

- Dynamically selecting authentication strategies based on real-time scene analysis.
- Integrating multimodal sensing (visual, audio, vehicle bus data).
- Using a Z3 solver for fuzzy logic reasoning and decision making.
- Supporting adaptive security strategies for Vehicles (V2V), Infrastructure (V2I), and Pedestrian Devices (V2P).

## 🧩 Features

- 🖼️ Visual scene classification via Vision-Language Model (Gemma3:27B).
- 🎙️ Environmental sound classification with CAMPPlus-Fbank.
- 🚌 Vehicle bus data integration (speed, throttle, brake, angle).
- 🧠 Z3-based fuzzy reasoning engine for strategy selection.
- 📊 Dynamic trust scoring and policy adjustment.
- 🏎️ Real-time testing in CARLA simulation environment.

## 🏗️ Project Structure

![Structure](https://github.com/shutendoujiQAQ/IoV-auto-auth-system/blob/main/structure.png)

## 🚀 Getting Started

### 1. Environment Setup

- Python 3.9
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- CARLA Simulator v0.9.13 (local or remote GPU server recommended).

### 2. Data Preparation

- Download UrbanSound8K and emergency vehicle alarm datasets for audio module.
- Place datasets in `data/` directory.

### 3. Running the System

- Launch CARLA simulation:
  ```bash
  ./CarlaUE4.sh
  ```
- Start the main controller:
  ```bash
  srart.bat
  ```

## 🧪 Testing & Evaluation

- Accuracy of visual classification: 85.6% across typical traffic scenarios.
- Audio classification accuracy: 94.3% (UrbanSound8K test set).
- Reasoning latency: <12ms per inference (Z3 Solver).
- End-to-end system latency: ~300ms.

## 📈 Roadmap

- Model distillation for on-vehicle deployment.
- Neuro-symbolic fusion reasoning.
- Self-learning rule optimization.
- Real-world dataset collection & testing.

## 📚 References

- [CARLA Simulator](https://carla.org/)
- Z3 SMT Solver: https://github.com/Z3Prover/z3

