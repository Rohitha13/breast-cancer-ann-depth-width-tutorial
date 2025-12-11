# 🧠 Breast Cancer Classification with ANN: Depth & Width Effects

## 📌 Project Overview
This tutorial explores how **Artificial Neural Network (ANN) architecture** - specifically **depth and width** - affects performance and overfitting in binary classification tasks. Using the Breast Cancer Wisconsin dataset, we demonstrate the trade-off between model capacity and generalization through three different ANN architectures.

## 🎯 Learning Goals
By completing this tutorial, you will learn to:
- Build and train ANNs using Keras/TensorFlow for tabular classification
- Interpret training/validation curves to detect overfitting
- Choose appropriate network architectures for different datasets
- Understand the relationship between model complexity and generalization

## 📊 Dataset
**Breast Cancer Wisconsin (Diagnostic) Dataset**
- **Source:** UCI Machine Learning Repository
- **Samples:** 569
- **Features:** 30 numeric measurements from digitized breast mass images
- **Target:** Diagnosis (M = malignant, B = benign)
- **Perfect for this tutorial:** Clean tabular data, medically meaningful, binary classification

## 🏗️ ANN Architectures Tested
We compare three configurations to illustrate the depth-width trade-off:

| Architecture | Layers | Neurons | Description |
|--------------|--------|---------|-------------|
| **Shallow/Narrow** | 1 hidden layer | 16 neurons | Simple patterns, low capacity |
| **Deeper** | 2 hidden layers | 16 neurons each | Moderate complexity |
| **Wide/Deep** | 2 hidden layers | 64, 32 neurons | High capacity, risk of overfitting |

## 🛠️ Technologies Used
- Python 3.7+
- TensorFlow 2.x / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Jupyter Notebook

## 📁 Project Structure
```
breast-cancer-classification-ann-98/
│
├── breast-cancer-classification-ann.ipynb  # Main tutorial notebook
├── data.csv                                    # Dataset (from UCI)
├── README.md                                   # This file
└── LICENSE                                     # MIT License
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-classification-ann-98.git
cd breast-cancer-classification-ann-98
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn plotly jupyter
```

### 3. Run the Notebook
```bash
jupyter notebook breast-cancer-classification-ann-98.ipynb
```

## 📚 Tutorial Sections
The notebook follows this logical flow:

1. **Introduction & Background**
   - Why ANNs for tabular data?
   - MLP architecture explained

2. **Data Exploration**
   - Loading and examining the dataset
   - Feature distributions by diagnosis
   - Missing values and duplicates check

3. **Data Preprocessing**
   - Label encoding (M/B → 1/0)
   - Train/test split (80/20)
   - Robust scaling for numerical features

4. **Model Building**
   - Three ANN architectures defined
   - Training with EarlyStopping and ReduceLROnPlateau
   - Batch normalization and dropout techniques

5. **Results & Analysis**
   - Training/validation loss curves
   - Overfitting detection
   - Performance comparison across architectures
   - Confusion matrices and classification reports

## 🔍 Key Insights
- **Shallow networks** learn simple patterns but may underfit complex data
- **Deeper networks** capture more complexity but require careful regularization
- **Wide/deep networks** have high capacity but easily overfit small datasets
- The tutorial demonstrates **how to read learning curves** to diagnose model issues

## 📊 Expected Results
- All models achieve **high accuracy** (>95%) on this well-separated dataset
- The **wide/deep network shows clear overfitting** (validation loss diverges)
- The **deeper network (2×16)** provides the best balance of performance and generalization

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- University of Wisconsin for the Breast Cancer dataset
- UCI Machine Learning Repository
- TensorFlow/Keras development team
- The machine learning education community

## 📖 Further Reading
1. **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
2. "Understanding the difficulty of training deep feedforward neural networks" - Glorot & Bengio (2010)
3. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" - Srivastava et al. (2014)
4. TensorFlow Documentation: https://www.tensorflow.org/guide/keras/overview

## 📧 Contact
For questions or feedback, please open an issue in the GitHub repository.

---
*This tutorial is designed for educational purposes. Always consult medical professionals for real-world medical diagnosis.*
