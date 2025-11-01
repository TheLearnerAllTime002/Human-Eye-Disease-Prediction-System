import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="OCT Retinal Analysis Platform",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom Metric: F1 Score
# ---------------------------
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-8))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# ---------------------------
# Model Functions
# ---------------------------
def create_model():
    INPUT_SHAPE = (224, 224, 3)
    mobnet = tf.keras.applications.MobileNetV3Large(
        input_shape=INPUT_SHAPE,
        alpha=1.0,
        minimalistic=False,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        include_preprocessing=True,
    )
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=INPUT_SHAPE))
    model.add(mobnet)
    model.add(tf.keras.layers.Dense(units=4, activation="softmax"))
    
    return model

@st.cache_resource
def load_trained_model():
    try:
        if os.path.exists("Trained_Model.h5"):
            try:
                model = tf.keras.models.load_model(
                    "Trained_Model.h5",
                    custom_objects={"F1Score": F1Score},
                    compile=False
                )
                return model
            except:
                pass
        
        model = create_model()
        if os.path.exists("Trained_Model.h5"):
            model.load_weights("Trained_Model.h5")
        else:
            st.error("❌ No model file found")
            st.stop()
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def model_prediction(uploaded_file, model):
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
        
        predictions = model.predict(img_array, verbose=0)
        class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        
        label_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        label = class_labels[label_index]
        
        return label, confidence, predictions[0]
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None

# ---------------------------
# Main App
# ---------------------------

# Header
st.markdown("# **OCT Retinal Analysis Platform** 👁️")

# Sidebar
with st.sidebar:
    st.markdown("## **Navigation**")
    page = st.selectbox("Choose a section:", ["🏠 Home", "📊 Analysis", "📚 About Dataset"])

if page == "🏠 Home":
    st.markdown("""
    #### **Welcome to the Retinal OCT Analysis Platform**

    **Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

    ##### **Why OCT Matters**
    OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.

    ---

    #### **Key Features of the Platform**

    - **Automated Image Analysis**: Our platform uses state-of-the-art machine learning models to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
    - **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
    - **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.

    ---

    #### **Understanding Retinal Diseases through OCT**

    1. **Choroidal Neovascularization (CNV)**
       - Neovascular membrane with subretinal fluid
       
    2. **Diabetic Macular Edema (DME)**
       - Retinal thickening with intraretinal fluid
       
    3. **Drusen (Early AMD)**
       - Presence of multiple drusen deposits

    4. **Normal Retina**
       - Preserved foveal contour, absence of fluid or edema

    ---

    #### **Get Started**

    - **Upload OCT Images**: Navigate to the Analysis section to begin uploading your OCT scans.
    - **Explore Results**: View categorized scans and detailed diagnostic insights.
    - **Learn More**: Check the About Dataset section to understand our comprehensive dataset.
    """)

elif page == "📊 Analysis":
    st.markdown("## **OCT Image Analysis**")
    st.write("Upload your OCT scan below for automated analysis:")
    
    # Load model
    model = load_trained_model()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an OCT image file", 
            type=["jpg", "jpeg", "png"],
            help="Upload a high-quality OCT scan in JPG, JPEG, or PNG format"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded OCT Scan", use_container_width=True)
    # rest of the analysis code
    
    with col2:
        if uploaded_file is not None:
            with st.spinner("🔍 Analyzing OCT scan..."):
                label, confidence, all_predictions = model_prediction(uploaded_file, model)
            
            if label is not None:
                # Main prediction
                st.markdown("### **Analysis Results**")
                
                if label == "NORMAL":
                    st.success(f"🟢 **Diagnosis:** {label}")
                elif label == "CNV":
                    st.error(f"🔴 **Diagnosis:** Choroidal Neovascularization (CNV)")
                elif label == "DME":
                    st.warning(f"🟡 **Diagnosis:** Diabetic Macular Edema (DME)")
                else:  # DRUSEN
                    st.info(f"🔵 **Diagnosis:** Drusen (Early AMD)")
                
                st.metric("Confidence Score", f"{confidence:.1%}")
                
                # Confidence interpretation
                if confidence > 0.8:
                    st.success("✅ High confidence prediction")
                elif confidence > 0.6:
                    st.warning("⚠️ Moderate confidence prediction")
                else:
                    st.error("❌ Low confidence - recommend expert consultation")
                
                # All predictions
                st.markdown("### **Detailed Probabilities**")
                class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
                for i, (cls, prob) in enumerate(zip(class_labels, all_predictions)):
                    st.progress(float(prob), text=f"{cls}: {prob:.1%}")
                
                st.markdown("---")
                st.info("⚠️ **Disclaimer:** This tool is for educational purposes only and should not replace professional medical diagnosis.")

elif page == "📚 About Dataset":
    st.markdown("## **About the Dataset**")
    
    st.markdown("""
    Our dataset consists of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets, split into four primary categories:
    - **Normal** - Healthy retinal tissue
    - **CNV** - Choroidal Neovascularization  
    - **DME** - Diabetic Macular Edema
    - **Drusen** - Early Age-related Macular Degeneration

    Each image has undergone multiple layers of expert verification to ensure accuracy in disease classification. The images were obtained from various renowned medical centers worldwide and span across a diverse patient population, ensuring comprehensive coverage of different retinal conditions.

    ---

    #### **Dataset Details**
    
    Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.

    **(A)** (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). (Middle right) Multiple drusen (arrowheads) present in early AMD. (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

    ---

    #### **Content**
    The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). There are 84,495 OCT images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

    Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

    Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People's Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.

    Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing expertise for verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The first tier of graders consisted of undergraduate and medical students who had taken and passed an OCT interpretation course review. This first tier of graders conducted initial quality control and excluded OCT images containing severe artifacts or significant image resolution reductions. The second tier of graders consisted of four ophthalmologists who independently graded each image that had passed the first tier. The presence or absence of choroidal neovascularization (active or in the form of subretinal fibrosis), macular edema, drusen, and other pathologies visible on the OCT scan were recorded. Finally, a third tier of two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image.
    """)

# Footer
st.markdown("---")
st.markdown("**Developed by Arjun** | Powered by TensorFlow & Streamlit 🚀")


