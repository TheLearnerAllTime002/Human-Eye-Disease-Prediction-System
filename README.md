# OCT Retinal Image Classification using Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oct-analyzer-ai.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

This research presents an automated diagnostic system for retinal disease classification using Optical Coherence Tomography (OCT) imaging. Leveraging transfer learning with MobileNetV3 architecture, the system achieves robust performance in multi-class classification of retinal pathologies including Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), Drusen, and Normal retinal tissue. The web-based deployment facilitates real-time inference, making advanced diagnostic capabilities accessible to healthcare professionals.

## 1. Introduction

Optical Coherence Tomography (OCT) has revolutionized ophthalmic diagnostics by providing non-invasive, high-resolution cross-sectional imaging of retinal structures. With over 30 million OCT scans performed annually worldwide, the volume of imaging data necessitates automated analysis systems to augment clinical decision-making and reduce diagnostic latency.

### 1.1 Clinical Significance

Retinal diseases represent a leading cause of visual impairment globally:
- **Choroidal Neovascularization (CNV)**: Abnormal blood vessel growth beneath the retina, associated with wet age-related macular degeneration
- **Diabetic Macular Edema (DME)**: Fluid accumulation in the macula due to diabetic retinopathy
- **Drusen**: Extracellular deposits indicative of early-stage age-related macular degeneration

## 2. Research Objectives

1. Develop a deep learning model for automated classification of OCT retinal images
2. Implement transfer learning to optimize model performance with limited medical imaging data
3. Deploy an accessible web-based platform for real-time inference
4. Provide interpretable predictions with confidence metrics for clinical validation

## 3. Methodology

### 3.1 Dataset

- **Source**: Kermany et al. OCT retinal imaging dataset
- **Classes**: 4 (CNV, DME, Drusen, Normal)
- **Total Images**: 84,495 images
- **Split Ratio**: 80% training, 10% validation, 10% testing
- **Image Specifications**: Grayscale, variable dimensions (normalized to 224×224)

### 3.2 Model Architecture

**Base Model**: MobileNetV3-Large (ImageNet pre-trained)
- Efficient convolutional neural network optimized for mobile and edge devices
- Transfer learning approach with frozen base layers
- Custom classification head:
  - Global Average Pooling
  - Dense layer (128 units, ReLU activation)
  - Dropout (0.5) for regularization
  - Output layer (4 units, Softmax activation)

### 3.3 Training Configuration

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Epochs**: 25 (with early stopping)
- **Batch Size**: 32
- **Data Augmentation**: Rotation, zoom, horizontal flip, brightness adjustment

### 3.4 Deployment Stack

- **Framework**: Streamlit (interactive web application)
- **Backend**: TensorFlow/Keras
- **Hosting**: Streamlit Community Cloud
- **Interface**: Responsive design with real-time image processing

## 4. Results

### 4.1 Model Performance

| Metric | CNV | DME | Drusen | Normal | Overall |
|--------|-----|-----|--------|--------|--------|
| **Precision** | 0.95 | 0.93 | 0.91 | 0.97 | 0.94 |
| **Recall** | 0.94 | 0.92 | 0.90 | 0.96 | 0.93 |
| **F1-Score** | 0.945 | 0.925 | 0.905 | 0.965 | 0.935 |
| **Accuracy** | - | - | - | - | **93.8%** |

### 4.2 Inference Performance

- **Average Prediction Time**: < 2 seconds per image
- **Model Size**: 15.2 MB (optimized for deployment)
- **Supported Image Formats**: JPEG, PNG, BMP

### 4.3 Sample Output

```
[Placeholder for sample prediction visualization]

Expected format:
- Input OCT scan image
- Predicted class with confidence score
- Probability distribution across all classes
- Visual attention heatmap (future enhancement)
```

![Sample OCT Analysis](sample_output_placeholder.png)
*Figure 1: Representative OCT scan with model prediction and confidence metrics*

## 5. Key Features

- ✅ **Automated Classification**: End-to-end inference pipeline
- ✅ **Real-time Processing**: Instant predictions via web interface
- ✅ **Probabilistic Outputs**: Complete confidence distribution across classes
- ✅ **Educational Content**: Integrated disease information and clinical context
- ✅ **Responsive Design**: Cross-platform compatibility (desktop, tablet, mobile)
- ✅ **Scalable Architecture**: Cloud-based deployment for accessibility

## 6. Installation & Usage

### 6.1 Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM (minimum)
- Modern web browser

### 6.2 Local Installation

```bash
# Clone the repository
git clone https://github.com/TheLearnerAllTime002/Human-Eye-Disease-Prediction-System.git
cd Human-Eye-Disease-Prediction-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify model files
# Ensure Trained_Model.h5 or Trained_Model.keras is present

# Run the application
streamlit run app.py
```

The application will launch at `http://localhost:8501`

### 6.3 Web Deployment

Access the live application: **[https://oct-analyzer-ai.streamlit.app](https://oct-analyzer-ai.streamlit.app)**

### 6.4 Usage Workflow

1. Navigate to the **Analysis** page
2. Upload an OCT retinal scan (JPEG/PNG format)
3. Click **Analyze Image**
4. Review prediction results and confidence scores
5. Consult disease information for clinical context

## 7. Dependencies

```
tensorflow==2.15.0
streamlit==1.31.1
numpy>=1.24.0
Pillow>=10.0.0
pandas>=2.0.0
matplotlib>=3.7.0
```

## 8. Future Enhancements

- [ ] Implement Grad-CAM for visual explanation of predictions
- [ ] Expand classification to additional retinal pathologies
- [ ] Integrate DICOM format support for clinical compatibility
- [ ] Develop longitudinal analysis for disease progression tracking
- [ ] Incorporate ensemble methods for improved robustness
- [ ] Add multi-language support for global accessibility

## 9. Limitations

- Model trained on specific dataset; generalization to diverse populations requires validation
- Not intended as a replacement for professional medical diagnosis
- Performance may vary with image quality and acquisition protocols
- Requires further clinical validation studies for regulatory approval

## 10. Citation

If you use this work in your research, please cite:

```bibtex
@misc{mitra2024oct,
  author = {Mitra, Arjun},
  title = {OCT Retinal Image Classification using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/TheLearnerAllTime002/Human-Eye-Disease-Prediction-System}},
  note = {Accessed: 2024}
}
```

### Dataset Citation

```bibtex
@article{kermany2018identifying,
  title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
  author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and others},
  journal={Cell},
  volume={172},
  number={5},
  pages={1122--1131},
  year={2018},
  publisher={Elsevier}
}
```

## 11. Author

**Arjun Mitra**  
Data Science & Machine Learning Researcher  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/arjun-mitra-2761a9260/)  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/TheLearnerAllTime002)

**Profile**: Specializing in computer vision applications for medical imaging, with focus on developing accessible AI-driven diagnostic tools for healthcare.

## 12. Contact

For questions, collaborations, or feedback:

- **LinkedIn**: [Arjun Mitra](https://www.linkedin.com/in/arjun-mitra-2761a9260/)
- **GitHub Issues**: [Report bugs or request features](https://github.com/TheLearnerAllTime002/Human-Eye-Disease-Prediction-System/issues)
- **Live Application**: [https://oct-analyzer-ai.streamlit.app](https://oct-analyzer-ai.streamlit.app)

## 13. Acknowledgements

This research acknowledges:

- **Dataset Providers**: Kermany et al. for the comprehensive OCT imaging dataset
- **TensorFlow Team**: For the robust deep learning framework
- **Streamlit**: For the intuitive web application development platform
- **Open Source Community**: For pre-trained models and collaborative tools
- **Medical Professionals**: For domain expertise and validation insights

## 14. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This application is designed for research and educational purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

---

*Last Updated: November 2024*  
*Version: 2.0*
