# Age and Gender Estimation with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive CNN-based system for age and gender estimation using ResNet50 transfer learning. This project includes an interactive Streamlit web application with real-time predictions, model training pipeline, and Grad-CAM interpretability analysis.

## 🎯 Key Features

- **ResNet50 Transfer Learning**: Efficient CNN architecture with frozen base layers (only 2.3% trainable parameters)
- **Dual-Task Learning**: Simultaneous age regression and gender classification
- **Model Interpretability**: Grad-CAM visualizations to understand model decision-making
- **Interactive Web App**: Complete Streamlit interface for training, evaluation, and predictions
- **Comprehensive Evaluation**: MAE analysis, confusion matrices, and performance visualization
- **Real-time Predictions**: Upload images for instant age and gender predictions

## 📊 Performance Results

- **Age Estimation**: 15.2 ± 18.5 years Mean Absolute Error
- **Gender Classification**: 72.5% accuracy with balanced precision/recall
- **Age Group Performance**: 
  - Young (0-25): 12.1 years MAE
  - Adult (26-50): 16.8 years MAE  
  - Senior (51+): 18.7 years MAE
- **Model Efficiency**: 8 epochs training, 20 minutes on CPU

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/age-gender-estimation-plastic-surgery.git
cd age-gender-estimation-plastic-surgery
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download UTKFace Dataset:**
```bash
# Option 1: Use our download script
python download_dataset.py

# Option 2: Manual download from official source
# Download from: https://susanqq.github.io/UTKFace/
# Extract to ./UTKFace/ directory
```

4. **Run the Streamlit application:**
```bash
streamlit run app.py --server.port 5000
```

5. **Access the application:**
Open your browser to `http://localhost:5000`

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── model_utils.py                  # Core model and data processing functions
├── gradcam.py                     # Grad-CAM implementation for interpretability
├── download_dataset.py            # Automatic dataset download utility
├── generate_results.py            # Results generation and analysis
├── large_dataset_loader.py        # Optimized data loading for large datasets
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT license
└── UTKFace/                       # Dataset directory (download required)
```

## 🎮 Using the Application

### 1. Dataset Overview
- View demographic distribution and age statistics
- Explore sample images from the UTKFace dataset
- Analyze dataset balance and characteristics

### 2. Model Training
- Configure training parameters (batch size, epochs, learning rate)
- Monitor real-time training progress with live metrics
- View loss curves and accuracy plots during training

### 3. Model Evaluation
- Comprehensive performance analysis with MAE and accuracy metrics
- Age group-specific performance breakdown
- Confusion matrix visualization for gender classification

### 4. Grad-CAM Visualization
- Interactive model interpretability analysis
- Visualize which facial regions influence predictions
- Understand model decision-making process

### 5. Interactive Prediction
- Upload custom images for real-time age and gender prediction
- View prediction confidence and attention maps
- Export results and visualizations

## 🧠 Model Architecture

The system uses a **ResNet50 transfer learning** approach with the following architecture:

| Layer | Output Shape | Parameters | Type |
|-------|--------------|------------|------|
| ResNet50 Base | (7, 7, 2048) | 23,587,712 | Feature Extractor (Frozen) |
| GlobalAveragePooling2D | (2048,) | 0 | Pooling |
| Dense (Feature) | (256,) | 524,544 | Feature Processing |
| Dense (Classifier) | (128,) | 32,896 | Classification |
| Age Output | (1,) | 129 | Regression Head |
| Gender Output | (2,) | 258 | Classification Head |
| **Total** | - | **24,145,539** | - |
| **Trainable** | - | **557,827** | **2.3% of total** |

## 🎯 Applications

This system can be used for various applications:

- **Age Estimation**: Accurate age prediction from facial images
- **Gender Classification**: Reliable gender classification with confidence scores
- **Computer Vision Research**: Baseline for age and gender estimation research
- **Educational Purposes**: Learning transfer learning and deep learning concepts
- **Model Interpretability**: Understanding CNN decision-making through Grad-CAM

## 📊 Analysis and Results

The system includes comprehensive analysis tools:

- **Performance Metrics**: Age MAE, gender accuracy, and detailed evaluation
- **Visualization Tools**: Training curves, confusion matrices, and prediction samples
- **Model Interpretation**: Grad-CAM analysis for understanding predictions
- **Results Generation**: Automated scripts for generating performance reports

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{age_gender_estimation_2025,
  title={Age and Gender Estimation with Deep Learning: A ResNet50 Transfer Learning Approach},
  author={[Your Name]},
  year={2025},
  howpublished={GitHub Repository},
  url={https://github.com/yourusername/age-gender-estimation}
}
```

## 🙏 Acknowledgments

This project builds upon and acknowledges the following important contributions:

### 📊 Dataset
- **UTKFace Dataset**: Zhang, Zhifei, et al. "Age progression/regression by conditional adversarial autoencoder." *IEEE Conference on Computer Vision and Pattern Recognition*. 2017.
  - Dataset Homepage: https://susanqq.github.io/UTKFace/
  - Used under the dataset's original license terms

### 💡 Original Inspiration
- **Kaggle Notebook**: "CNN-Based Age Gender Estimation" by Amerhu
  - Original notebook: https://www.kaggle.com/code/amerhu/cnn-based-age-gender-estimation/notebook
  - This project extends and enhances the original work with comprehensive analysis and improved features

### 🔧 Technical Foundations
- **ResNet50**: He, Kaiming, et al. "Deep residual learning for image recognition." *CVPR* 2016.
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **Grad-CAM**: Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." *ICCV* 2017.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### License Summary:
- ✅ Commercial use allowed
- ✅ Modification allowed  
- ✅ Distribution allowed
- ✅ Private use allowed
- ❗ Must include license and copyright notice
- ❗ No warranty provided

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Issues & Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/age-gender-estimation-plastic-surgery/issues) page
2. Create a new issue with detailed description
3. Include error messages, system information, and steps to reproduce

## 🔗 Related Work

- [Original UTKFace Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Age_ProgressionRegression_by_CVPR_2017_paper.pdf)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Facial Age Estimation Survey](https://ieeexplore.ieee.org/document/8454919)

## 📊 Performance Benchmarks

Detailed performance comparisons and benchmarks can be reproduced using the provided evaluation scripts.

---

**Made with ❤️ for the computer vision and machine learning communities**

*Last updated: July 2025*