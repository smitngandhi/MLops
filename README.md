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
