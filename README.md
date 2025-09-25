# 🏥 UM Internship Projects - Medical AI Prediction Suite

A comprehensive collection of machine learning applications for medical diagnosis and prognosis prediction, developed during the University of Miami internship program.

## 📋 Project Overview

This repository contains **4 AI prediction applications** built using machine learning techniques to assist healthcare professionals in diagnosis and prognosis assessment. Each project includes both training scripts and standalone GUI applications compiled as Windows executables.

## 🚀 Completed Projects

### 1. ❤️ **Thyroid Cancer Recurrence Prediction**
- **Folder**: `Thyroid_Cancer_Prediction/`
- **Model**: Random Forest Classifier
- **Accuracy**: 96% (optimized with parameter tuning)
- **Features**: 16 clinical parameters including TNM staging, pathology, demographics
- **Executable**: `Thyroid_Cancer_Predictor.exe` (232.6 MB)
- **Status**: ✅ **Production Ready**

### 2. 🫀 **Heart Disease Risk Assessment**
- **Folder**: `Heart_Disease_Prediction/`
- **Model**: Random Forest Classifier  
- **Accuracy**: High performance with balanced dataset
- **Features**: 13 cardiovascular risk factors
- **Executable**: `Heart_Disease_Predictor.exe` (165.3 MB)
- **Status**: ✅ **Production Ready**

### 3. ❤️‍🩹 **Liver Cirrhosis Stage Classification**
- **Folder**: `Liver_Cirrhosis_Stage_Prediction/`
- **Model**: XGBoost (95.8% F1-Score) with Random Forest fallback
- **Accuracy**: 95.8% F1-Score, 95.32% test accuracy
- **Features**: 18 clinical and laboratory parameters (11 numerical, 7 categorical)
- **GUI Status**: ✅ **Fully Functional** (prediction errors resolved)
- **Core ML**: ✅ **Complete** with comprehensive evaluation metrics
- **Status**: 🔧 **Core Ready** (building disabled, functionality perfected)

### 4. 😺 **Animal Classification (Dual-Stage AI)**
- **Folder**: `Animal_Classification/`
- **Model**: Dual-Stage EfficientNetV2B0 Architecture (Stage 1 + Stage 2)
- **Accuracy**: High precision with intelligent prediction consolidation
- **Features**: 15 animal classes with advanced ensemble prediction
- **Executable**: `Animal_Classification_Dual_Stage.exe` (577.5 MB)
- **Status**: ✅ **Production Ready**

## 🛠 Technology Stack

### **Core Libraries**
- **scikit-learn**: Machine learning algorithms and preprocessing
- **tensorflow**: Neural Networks and deep learning
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization
- **joblib**: Model serialization

### **GUI Framework**
- **PyQt5**: Professional medical-grade user interfaces
- **Dark theme**: Modern, clinical appearance suitable for healthcare settings

### **Deployment**
- **PyInstaller**: Standalone executable compilation
- **Windows compatible**: No Python installation required on target machines

### **Machine Learning Techniques**
- **Random Forest**: Primary algorithm for stability and interpretability
- **Neural Networks**: Primary algorithm for image classification
- **Parameter Tuning**: GridSearchCV for optimal hyperparameters
- **Class Balancing**: SMOTE and class weights for imbalanced datasets
- **Feature Engineering**: Domain-specific medical feature creation
- **Pipeline Architecture**: Integrated preprocessing and modeling

## 📊 Performance Metrics

| Project | Model | Accuracy | GUI Status | Core ML Status |
|---------|--------|----------|------------|----------------|
| Thyroid Cancer | Random Forest | 96%+ | ✅ Production Ready | ✅ Complete |
| Heart Disease | Random Forest | High | ✅ Production Ready | ✅ Complete |
| Liver Cirrhosis | XGBoost/RandomForest | 95.8% F1 | ✅ Fully Functional | ✅ Complete |
| Animal Classification | Dual-Stage EfficientNetV2B0 | High | ✅ Production Ready | ✅ Complete |

## 🏗 Project Structure

```
UM Internship Projects/
│
├── thyroid_cancer/
│   ├── main.py                              # Model training script
│   ├── thyroid_cancer_app.py               # GUI application
│   ├── build.py                            # Executable build script
│   ├── dataset.csv                         # Training data
│   ├── random_forest_thyroid_cancer_model.pkl
│   └── dist/Thyroid_Cancer_Predictor.exe
│
├── Heart Disease/
│   ├── main.py                              # Model training script
│   ├── heart_disease_app.py                # GUI application
│   ├── build.py                            # Executable build script
│   ├── dataset.csv                         # Training data
│   ├── heart_disease_model.pkl
│   ├── heart_disease_scaler.pkl
│   └── dist/Heart_Disease_Predictor.exe
│
├── Liver_Cirrhosis_Stage_Prediction/
│   ├── liver_cirrhosis_main.py             # Enhanced model training script
│   ├── liver_cirrhosis_app.py              # Professional PyQt5 GUI application
│   ├── test_values_and_models.py           # Comprehensive testing suite
│   ├── debug_prediction.py                 # Prediction debugging tools
│   ├── build.py & minimal_build.py         # Executable build scripts
│   ├── liver_cirrhosis.csv                 # Training dataset (25,000 samples)
│   ├── best_liver_cirrhosis_model.pkl      # XGBoost pipeline (95.8% F1)
│   ├── random_forest_liver_cirrhosis_model.pkl # Random Forest fallback
│   ├── preprocessor.pkl                     # Full preprocessing pipeline
│   ├── evaluation_metrics.png              # Comprehensive performance plots
│   └── requirements.txt                     # Dependencies list
│
├── Animal_Classification/
│   ├── main.py                              # Model training script
│   ├── animal_classification_app.py         # Dual-stage GUI application
│   ├── build.py                            # Executable build script
│   ├── aniClass_EFF_Stage1.pkl             # Stage 1 EfficientNet model
│   ├── aniClass_EFF_Stage2.pkl             # Stage 2 EfficientNet model
│   ├── class_names.json                    # Animal class definitions
│   └── dist/Animal_Classification_Dual_Stage.exe
│
└── README.md                               # This file
```

## 🚀 Quick Start Guide

### **For End Users**
1. Navigate to any project's `dist/` folder (or run `.py` apps directly)
2. Double-click the `.exe` file to launch the application
3. Enter patient parameters in the GUI
4. Click "Predict" to get risk assessment
5. No software installation required for executables!

### **Testing Liver Cirrhosis Application**
Use these **validated test values** in the GUI:
```
N_Days: 500        | Age: 45           | Sex: F
Status: C          | Drug: Placebo     | Ascites: N
Hepatomegaly: N    | Spiders: N        | Edema: N
Bilirubin: 1.2     | Cholesterol: 250  | Albumin: 3.8
Copper: 80         | Alk_Phos: 1200    | SGOT: 120
Tryglicerides: 150 | Platelets: 300    | Prothrombin: 11.0
```
**Expected Result**: Cirrhosis stage prediction with confidence percentages

#### **Prerequisites**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn PyQt5 joblib imbalanced-learn tensorflow
```

#### **Training a Model**
```bash
cd "project_folder"
python main.py
```

#### **Running GUI Application**
```bash
python project_app.py
```

#### **Building Executable**
```bash
python build.py
```

## 🎯 Key Features

### **Medical-Grade Applications**
- ✅ **Professional GUI**: Clean, clinical interface design
- ✅ **Input Validation**: Prevents invalid medical parameter entries
- ✅ **Real-time Predictions**: Instant risk assessment
- ✅ **Color-coded Results**: Visual risk indication (Green/Yellow/Red)
- ✅ **Probability Scores**: Exact percentage likelihood
- ✅ **Error Handling**: Robust error messages and recovery

### **Advanced ML Pipeline**
- ✅ **Feature Engineering**: Medical domain-specific feature creation
- ✅ **Data Preprocessing**: Standardization, encoding, imputation
- ✅ **Class Balancing**: SMOTE and weighted algorithms for imbalanced data
- ✅ **Model Optimization**: GridSearchCV parameter tuning
- ✅ **Cross-validation**: Robust performance evaluation
- ✅ **Feature Importance**: Clinical insights into prediction factors

### **Deployment Ready**
- ✅ **Standalone Executables**: No Python installation required
- ✅ **Windows Compatible**: Runs on Windows 7/8/10/11
- ✅ **Portable**: Single file deployment
- ✅ **Professional Packaging**: Medical software standards

## 📈 Clinical Applications

### **Thyroid Cancer Application**
- **Use Case**: Post-surgical recurrence risk assessment
- **Target Users**: Endocrinologists, oncologists
- **Key Parameters**: TNM staging, pathology type, patient demographics
- **Clinical Value**: Treatment planning and follow-up scheduling

### **Heart Disease Application**
- **Use Case**: Cardiovascular risk screening
- **Target Users**: Cardiologists, primary care physicians
- **Key Parameters**: Blood pressure, cholesterol, ECG results, lifestyle factors
- **Clinical Value**: Early intervention and prevention strategies

### **Liver Cirrhosis Application**
- **Use Case**: Disease staging and prognosis assessment (3 stages: Early, Moderate, Advanced)
- **Target Users**: Hepatologists, gastroenterologists, liver specialists
- **Key Parameters**: 18 clinical features including bilirubin, albumin, ascites, copper levels, liver enzymes
- **Clinical Value**: Treatment selection, liver transplant evaluation, disease monitoring
- **Technical Features**: SHAP explainability, confidence scoring, risk factor analysis
- **Current Status**: ✅ GUI fully functional, predictions working correctly with comprehensive test suite

### **Animal Classification Application**
- **Use Case**: Computer vision research and educational demonstrations
- **Target Users**: Researchers, educators, students
- **Key Parameters**: Image processing with dual-stage AI architecture
- **Technical Value**: Advanced ensemble learning with EfficientNet models

## 🔬 Research & Development Notes

### **Recent Updates (September 2025)**
- ✅ **Liver Cirrhosis Project Enhanced**: Fixed prediction errors, improved data type handling
- ✅ **Comprehensive Testing Suite**: Added test_values_and_models.py with realistic medical test cases
- ✅ **Debug Tools Created**: debug_prediction.py for troubleshooting prediction functionality
- ✅ **GUI Stability Improved**: Resolved "ufunc 'isnan' not supported" errors through robust type conversion
- ✅ **Model Performance Validated**: XGBoost achieving 95.8% F1-Score with proper evaluation metrics

### **Algorithm Selection Rationale**
- **XGBoost selected** for Liver Cirrhosis (95.8% F1-Score) with Random Forest fallback
- **Random Forest chosen** for other projects due to PyInstaller compatibility
- **Ensemble methods** provide robust predictions with medical data
- **Interpretability** important for clinical decision support
- **Class balancing** critical for medical datasets with rare positive outcomes

### **Data Quality Considerations**
- **Missing value handling**: Robust imputation strategies
- **Outlier detection**: Medical parameter validation
- **Feature scaling**: Standardization for mixed clinical parameters
- **Cross-validation**: Stratified sampling for medical outcomes

### **Deployment Challenges Solved**
- **Dependency management**: sklearn/scipy version compatibility
- **File path resolution**: PyInstaller temporary directory handling
- **GUI threading**: Responsive interface during predictions
- **Error handling**: Medical application reliability requirements
- **Data type conversion**: Fixed mixed GUI input types causing prediction errors
- **XGBoost compatibility**: Resolved PyInstaller conflicts while maintaining model performance
- **Prediction pipeline**: Robust preprocessing with proper NaN/infinite value handling

### **Short Term (Current Sprint)**
- 📱 **Mobile Compatibility**: Explore cross-platform deployment
- 🔍 **Model Explainability**: Add SHAP values for clinical interpretability

### **Medium Term**
- 🧠 **Deep Learning Integration**: Neural networks for complex medical patterns
- 📊 **Real-time Monitoring**: Integration with electronic health records
- 🌐 **Web Application**: Browser-based versions for wider accessibility

### **Long Term**
- 🤖 **AI-Assisted Diagnosis**: Multi-modal medical data integration
- 🔗 **Clinical Decision Support**: Integration with hospital systems
- 📈 **Continuous Learning**: Model updates with new medical data

## 👥 Contributors

**Developed during Unified Mentor Internship Program**
- **Intern**:  Sidhartha Das
- **Institution**: Unified Mentor
- **Period**: Jun 2024 - Sep 2024

## 📄 License & Usage

**For Educational and Research Purposes**
- ⚠️ **Medical Disclaimer**: These applications are for research and educational purposes only
- ⚠️ **Not for Clinical Use**: Not approved for direct patient care without validation
- ⚠️ **Supervision Required**: Use under qualified medical professional supervision

## 🤝 Acknowledgments

- **Unified Mentor** for internship opportunity and datasets
- **Medical Faculty** for domain expertise and guidance
- **Open Source Community** for machine learning libraries and tools
- **Healthcare Professionals** for clinical insights and feedback

---