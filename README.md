# 📰 Advancing Fake News Detection: Hybrid Deep Learning with FastText and Explainable AI

## 🎯 Objective
This project aims to enhance the detection of fake news using a hybrid deep learning model that combines the power of **FastText** for word embedding and **deep learning architectures**. Additionally, we integrate **Explainable AI (XAI)** techniques to provide transparency into the decision-making process.

---

## 📂 Dataset
We use an open-source dataset consisting of labeled news articles classified as **FAKE** or **REAL**. The dataset includes fields such as:
- 🗞️ `title`: Headline of the news
- 📜 `text`: Full news content
- 🏷️ `label`: Classification as FAKE or REAL

👉 https://drive.google.com/drive/folders/13XORhO2wKbAX9MA-cBRvkcq5Mr3oEYOG?usp=sharing

---

## ⚙️ Steps Involved

### 1️⃣ Data Preprocessing
- Clean and normalize the text
- Remove stopwords and punctuation
- Tokenize the news content
- Apply **FastText embeddings** for vector representation

### 2️⃣ Model Building
- Build a **Hybrid Deep Learning Model** (e.g., CNN + LSTM or BiLSTM)
- Train on embedded vectors
- Apply dropout and regularization for generalization

### 3️⃣ Model Evaluation
- Evaluate with metrics like **Accuracy**, **Precision**, **Recall**, **F1-Score**
- Visualize results with confusion matrix and training graphs

### 4️⃣ Explainable AI
- Use XAI tools like **LIME**, **SHAP**, or **Attention Mechanisms**
- Visualize feature importance and model interpretability

### 5️⃣ Manual Testing
- Load saved model (`.h5`)
- Accept custom news inputs
- Predict and display result: ✅ REAL / ❌ FAKE

---

## 🛠️ Technologies Used
- 🐍 Python
- 🤖 TensorFlow / Keras
- 📊 Pandas, NumPy, Matplotlib, Seaborn
- 🧠 FastText (Gensim)
- 🧪 Scikit-learn
- 🔍 XAI Libraries: LIME / SHAP
- 📦 Jupyter Notebook

---

## 🗂️ Folder Structure
