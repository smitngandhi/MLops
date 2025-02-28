# **First MLOps Project**  

## **📌 Tutorial 1: Setting Up the Project**  

### **1️⃣ Create a GitHub Repository**  
- Initialize a new GitHub repository for the project.  

### **2️⃣ Set Up a Conda Environment**  
- Run the following command to create a Conda environment with Python 3.8:  
  ```bash
  conda create -p my_env python==3.8 -y
  ```
- Activate the environment:  
  ```bash
  conda activate venv/
  ```

### **3️⃣ Initialize Git & Push Code**  
- Initialize Git:  
  ```bash
  git init
  ```  
- Add a README file:  
  ```bash
  git add README.md
  git commit -m "Initial commit"
  ```  
- Push the code to GitHub.  

### **4️⃣ Set Up `.gitignore`**  
- Create a `.gitignore` file for Python and pull it into your local machine.  

### **5️⃣ Create Essential Project Files**  
Inside the project directory, create the following files:  
- `src/__init__.py`  
- `setup.py`  
- `requirements.txt`  

### **6️⃣ Install Dependencies**  
- Install all required dependencies using:  
  ```bash
  pip install -r requirements.txt
  ```  

### **7️⃣ Add Changes to Git**  
- Track and commit the new files:  
  ```bash
  git add .
  git commit -m "Added setup and dependencies"
  git push origin main
  ```

---

## **📌 Tutorial 2: Project Structure**  

### **1️⃣ Organizing the Codebase**  
Inside the `src/` directory, create the following subdirectories:  

#### **🔹 Components (Handles core ML tasks)**  
- `__init__.py`  
- `data_ingestion.py` (Handles data collection & preprocessing)  
- `data_transformation.py` (Performs feature engineering & transformations)  
- `model_training.py` (Trains and evaluates ML models)  

#### **🔹 Pipeline (Manages end-to-end workflow)**  
- `__init__.py`  
- `train_pipeline.py` (Triggers the full training pipeline)  
- `predict_pipeline.py` (Handles model inference)  

#### **🔹 Utility & Error Handling Modules**  
- `exception.py` (Handles custom exceptions)  
- `logger.py` (Manages logging & debugging)  
- `utils.py` (Contains helper functions)  

# **📌 Tutorial 3: Structuring a Machine Learning Project**

### **1️⃣ Project Setup**
- Create a **notebook** folder to organize your Jupyter notebooks.
- Inside the notebook folder, create a **data** folder to store your dataset.

### **2️⃣ Create Jupyter Notebooks**
- Inside the notebook folder, create two Jupyter notebooks:
  1. **EDA Notebook** → Perform exploratory data analysis.
  2. **Model Training Notebook** → Train and evaluate machine learning models.

### **3️⃣ Install Dependencies**
- Add the required libraries to `requirements.txt`:
  ```
  catboost  
  xgboost  
  scikit-learn  
  ```
- Install them using:
  ```bash
  pip install -r requirements.txt
  ```

### **4️⃣ Run Jupyter Notebooks**
- Open and execute the notebooks to verify the workflow.
- Conduct data exploration in the **EDA notebook**.
- Train and evaluate models in the **Model Training notebook**.

### **5️⃣ Modular Coding Approach**
- Refactor the codebase for better maintainability:
  - Define **data preprocessing functions**.
  - Create a separate **module for model training**.
  - Structure code for **reusability and readability**.

### **6️⃣ Version Control with Git**
- Commit and push all updated files to GitHub.
- Organize them into separate sections:
  - **EDA**
  - **Problem Statements**

=======
 
  # **📌 Tutorial 4: Data Ingestion**

## **1️⃣ Data Ingestion Configuration**
- A `DataIngestionConfig` dataclass is created to define file paths for storing raw, train, and test data inside an `artifacts/` directory.
- Paths include:
  - `artifacts/data.csv` → Raw dataset
  - `artifacts/train.csv` → Training dataset
  - `artifacts/test.csv` → Testing dataset

## **2️⃣ Implementing the Data Ingestion Class**
- A `DataIngestion` class is created with:
  - A constructor (`__init__`) to initialize the configuration.
  - A method `initiate_data_ingestion()` to load, split, and store the dataset.

## **3️⃣ Data Ingestion Workflow**
The `initiate_data_ingestion()` method performs the following steps:

### **🔹 Step 1: Load the Dataset**
- Reads the dataset from `notebook/data/stud.csv` using `pandas`.
- Logs the successful loading of the dataset.

### **🔹 Step 2: Create Artifacts Directory**
- Uses `os.makedirs()` to ensure the `artifacts/` directory exists.
- Logs the creation of the directory.

### **🔹 Step 3: Save Raw Data**
- Stores the loaded dataset as `data.csv` inside `artifacts/`.
- Logs the saving of raw data.

### **🔹 Step 4: Train-Test Split**
- Splits the dataset into training (80%) and testing (20%) sets using `train_test_split()`.
- Saves the split data as `train.csv` and `test.csv`.
- Logs the successful completion of data ingestion.

## **4️⃣ Exception Handling & Logging**
- A `try-except` block is implemented to handle errors.
- Custom exceptions are raised using `CustomException` from `src.exception`.
- Logging is performed using `src.logger` to track progress and issues.

## **5️⃣ Running the Data Ingestion Pipeline**
- The script runs the data ingestion process when executed as the main script:
  ```python
  if __name__ == "__main__":
      obj = DataIngestion()
      obj.initiate_data_ingestion()
  ```
- This triggers the ingestion pipeline, ensuring data is correctly processed and stored.

<<<<<<< HEAD

  # **📌 Tutorial 6: Data Transformation**

## **1️⃣ Overview**
Data transformation is a crucial step in any machine learning pipeline. It involves handling missing values, encoding categorical variables, scaling numerical features, and preparing the dataset for training.

---

## **2️⃣ Data Transformation Pipeline**
The `DataTransformation` class is responsible for:
- Handling missing values in numerical and categorical features.
- Encoding categorical variables using OneHotEncoder.
- Scaling numerical and categorical features.
- Saving the preprocessor object for future use.

---

## **3️⃣ Implementation of `DataTransformation` Class**

### **🔹 Importing Required Libraries**
```python
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
```

### **🔹 Configuration Class**
```python
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
```
This class defines the path where the preprocessor object will be saved.

### **🔹 Data Transformation Class**
```python
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
```
This initializes the `DataTransformationConfig` object.

### **🔹 Creating the Preprocessor Pipeline**
```python
    def get_data_transform_obj(self):
        try:
            logging.info("Data Transformation object generation started")

            numeric_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoding', OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipelines', num_pipeline, numeric_features),
                    ('categorical_pipelines', categorical_pipeline, categorical_features)
                ]
            )

            logging.info("Data Transformation object generation finished")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
```

### **🔹 Applying Transformation to Train and Test Data**
```python
    def initiate_data_transform(self, train_path, test_path):
        try:
            logging.info("Loading Dataset")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocess_obj = self.get_data_transform_obj()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Dataset loaded into train and test input and target features")

            input_feature_train_arr = preprocess_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocess_obj.transform(input_feature_test_df)

            logging.info("Transforming data completed")

            logging.info("Concatenating train and test data started")
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info("Concatenating train and test data completed")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocess_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
```
