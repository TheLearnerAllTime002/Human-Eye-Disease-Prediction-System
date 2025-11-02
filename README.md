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
  - Training: 83,484 images
  - Testing: 1,000 images (250 per class)
- **Image Format**: Grayscale, varying resolutions (normalized to 224x224)

### 3.2 Data Preprocessing
```python
preprocessing_pipeline = {
    'resize': (224, 224),
    'normalization': 'min-max [0,1]',
    'augmentation': [
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'horizontal_flip': True,
        'zoom_range': 0.2
    ]
}
```

### 3.3 Model Architecture: MobileNetV3-Large
**Transfer Learning Configuration:**
```python
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base layers initially

# Custom classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])
```

**Rationale for MobileNetV3:**
- Optimized for mobile and edge deployment
- Efficient architecture (lightweight, fast inference)
- Strong performance on medical imaging tasks
- Reduced computational requirements compared to VGG/ResNet variants

### 3.4 Training Strategy
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 20
- **Callbacks**: Early Stopping (patience=5), ModelCheckpoint
- **Class Weights**: Computed to handle potential imbalances

## 4. Results

### 4.1 Model Performance

**Overall Metrics:**
- **Overall Accuracy**: 98.8890%
- **Overall F1-Score**: 0.988850 (98.8850%)

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| CNV | 0.99 | 0.99 | 0.99 | 250 |
| DME | 0.98 | 1.00 | 0.99 | 250 |
| DRUSEN | 1.00 | 0.97 | 0.98 | 250 |
| NORMAL | 0.99 | 1.00 | 0.99 | 250 |

### 4.2 Confusion Matrix Analysis
![Confusion Matrix](https://github.com/TheLearnerAllTime002/Human-Eye-Disease-Prediction-System/blob/main/output.png)

**Key Observations:**
- Minimal confusion between disease classes
- Highest confusion: DRUSEN → CNV (8 misclassifications)
- No confusion between NORMAL and diseased states

**Training Insights:**
- Convergence achieved by epoch 15
- No significant overfitting observed
- Validation accuracy stabilized at ~98.9%

## 5. Web Application

### 5.1 Streamlit Deployment
**Live Application**: [oct-analyzer-ai.streamlit.app](https://oct-analyzer-ai.streamlit.app)

**Features:**
- Real-time OCT image upload and analysis
- Confidence score visualization
- Class probability distribution
- Diagnostic recommendations
- Responsive web interface

### 5.2 Technical Stack
```python
technology_stack = {
    'Frontend': 'Streamlit',
    'Backend': 'TensorFlow 2.15',
    'Model Serving': 'TensorFlow Serving',
    'Deployment': 'Streamlit Cloud',
    'Image Processing': 'OpenCV, PIL'
}
```

## 6. Installation & Usage

### 6.1 Local Setup
```bash
# Clone repository
git clone https://github.com/TheLearnerAllTime002/Human-Eye-Disease-Prediction-System.git
cd Human-Eye-Disease-Prediction-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Run Streamlit App
```bash
streamlit run app.py
```

### 6.3 Model Training (Optional)
```bash
python train_model.py --epochs 20 --batch_size 32 --learning_rate 0.001
```

## 7. Project Structure
```
Human-Eye-Disease-Prediction-System/
│
├── app.py                   # Streamlit web application
├── train_model.py           # Model training script
├── model.h5                 # Trained model weights
├── requirements.txt         # Python dependencies
│
├── data/
│   ├── train/              # Training images
│   └── test/               # Testing images
│
├── notebooks/
│   ├── EDA.ipynb           # Exploratory data analysis
│   └── model_training.ipynb # Training experiments
│
├── utils/
│   ├── preprocessing.py     # Data preprocessing utilities
│   └── visualization.py     # Plotting functions
│
└── docs/
    └── model_architecture.md
```

## 8. Key Technologies
- **TensorFlow 2.15**: Deep learning framework
- **Keras**: High-level neural networks API
- **Streamlit**: Web application framework
- **OpenCV**: Image processing library
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization libraries

## 9. Limitations & Future Work

### 9.1 Current Limitations
- Dataset limited to 4 classes (excludes other retinal pathologies)
- Model trained on specific imaging protocol (may not generalize to all OCT devices)
- Requires further clinical validation on diverse patient populations

### 9.2 Proposed Enhancements
1. **Multi-Disease Expansion**: Include additional retinal conditions (retinal detachment, glaucoma, etc.)
2. **Explainability**: Integrate Grad-CAM for visual explanations of model decisions
3. **Ensemble Methods**: Combine multiple architectures for improved robustness
4. **Edge Deployment**: Optimize model for mobile devices using TensorFlow Lite
5. **Clinical Integration**: Develop DICOM compatibility for integration with hospital systems

## 10. Citation

**Publication Status:**  
*Note: This project represents ongoing research and development work. No peer-reviewed publication or article has been produced from this project yet. The system and results described here are part of an independent research initiative.*

If you use this code or methodology in your research or projects, please cite:

```bibtex
@misc{mitra2024oct,
  author = {Arjun Mitra},
  title = {OCT Retinal Image Classification using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/TheLearnerAllTime002/Human-Eye-Disease-Prediction-System}
}
```

**APA Style:**
```
Mitra, A. (2024). OCT Retinal Image Classification using Deep Learning [Software]. GitHub. https://github.com/TheLearnerAllTime002/Human-Eye-Disease-Prediction-System
```

**IEEE Style:**
```
A. Mitra, "OCT Retinal Image Classification using Deep Learning," GitHub repository, 2024. [Online]. Available: https://github.com/TheLearnerAllTime002/Human-Eye-Disease-Prediction-System
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
