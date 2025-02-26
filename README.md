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

# **Tutorial-3: Structuring a Machine Learning Project**

## **Step 1: Project Setup**
- Create a **notebook folder** for organizing your Jupyter notebooks.
- Inside the notebook folder, create a **data folder** to store your dataset.

## **Step 2: Notebook Creation**
- Create two Jupyter notebooks inside the notebook folder:
  1. **EDA Notebook** → For exploratory data analysis.
  2. **Model Training Notebook** → For training machine learning models.

## **Step 3: Install Required Libraries**
- Add the following dependencies to `requirements.txt`:
  ```
  catboost  
  xgboost  
  scikit-learn  
  ```
- Install them using:
  ```bash
  pip install -r requirements.txt
  ```

## **Step 4: Run the Jupyter Notebooks**
- Open and execute the notebooks to ensure proper workflow.
- Perform data exploration in the **EDA notebook**.
- Train models in the **Model Training notebook**.

## **Step 5: Modular Coding Approach**
- Refactor the project using **modular coding techniques**:
  - Define **data preprocessing functions**.
  - Create a separate **module for model training**.
  - Ensure reusability by organizing functions properly.

## **Step 6: Version Control with Git**
- Upload all the updated files to GitHub.
- Organize them into two sections:
  - **EDA**
  - **Problem Statements**



