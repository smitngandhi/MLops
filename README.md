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

---


