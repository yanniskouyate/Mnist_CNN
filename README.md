# 📘 MNIST Classification with CNN (PyTorch)
🔍 Overview

This project implements a full deep learning pipeline to classify handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN) built with PyTorch.
It includes preprocessing, model training, evaluation, and Kaggle submission generation.

🧠 Model Architecture
	•	Conv2D(1 → 32) + ReLU
	•	MaxPool(2×2)
	•	Conv2D(32 → 64) + ReLU
	•	MaxPool(2×2)
	•	Flatten
	•	Linear(64×7×7 → 128) + ReLU
	•	Dropout(0.3)
	•	Linear(128 → 10)

Validation accuracy reaches ≈ 99%.

⸻
📊 Dataset

Dataset used (Kaggle):
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
	•	Training: 60000 × 785
	•	Column 0 → label
	•	Columns 1–784 → pixels
	•	Test: 10000 × 785
	•	Column 0 is removed (index), not a label
	•	Columns 1–784 → pixels only

All images are reshaped to (1, 28, 28) and normalized to [0,1].

⸻

⚙️ Training
	•	Optimizer: Adam
	•	Loss: CrossEntropyLoss
	•	Batch size: 64
	•	Epochs: 8
	•	GPU automatically used if available.

The notebook includes:
	•	Loss curves
	•	Accuracy curves
	•	Error analysis
	•	Confusion matrix

  ⸻

📝 Kaggle Submission

The notebook generates a submission file: "submission.csv"
  ⸻
▶️ How to Run

git clone https://github.com/yanniskouyate/mnist_cnn.git
cd mnist_cnn
pip install torch torchvision pandas numpy matplotlib seaborn
jupyter notebook Mnist_CNN.ipynb

⸻
📜 License

MIT License.

⸻

👤 Author

Yannis Kouyate
M1 MACIA – Applied Mathematics & AI
GitHub: yanniskouyate
Email: yannis.kouyate@gmail.com

