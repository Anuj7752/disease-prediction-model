# 🏥 MediPredict AI — Disease Prediction System

> An intelligent full-stack web application that predicts diseases based on user symptoms using Machine Learning.

![Tech Stack](https://img.shields.io/badge/Stack-MERN%20%2B%20Python-blue)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![React](https://img.shields.io/badge/React-18-61DAFB)

---

## 📌 About The Project

**MediPredict AI** is a disease prediction web application built using the MERN stack (MongoDB, Express.js, React.js, Node.js) integrated with a Python-based Machine Learning model.

Users can select their symptoms from a list and the system will predict the most likely disease along with a confidence percentage.

---

## ✨ Features

- 🔍 Predict diseases based on 27 symptoms
- 🧠 Machine Learning powered by Random Forest Classifier
- 📊 Confidence/accuracy percentage shown with prediction
- 🕓 Prediction history saved in MongoDB
- 🌐 Responsive dark-themed UI
- 🔄 Real-time API communication between all 3 services

---

## 🦠 Diseases It Can Predict (12 Total)

| # | Disease | # | Disease |
|---|---------|---|---------|
| 1 | Flu | 7 | Typhoid |
| 2 | COVID-19 | 8 | Pneumonia |
| 3 | Common Cold | 9 | Diabetes |
| 4 | Asthma | 10 | Migraine |
| 5 | Dengue | 11 | Gastroenteritis |
| 6 | Malaria | 12 | Hypertension |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React.js + Vite + Tailwind CSS |
| Backend | Node.js + Express.js |
| Database | MongoDB + JSON fallback |
| ML Model | Python + Flask |
| Algorithm | Random Forest Classifier |
| Libraries | scikit-learn, pandas, joblib |

---

## 📁 Project Structure

```
disease-prediction/
├── client/          # React.js Frontend (Vite)
│   ├── src/
│   │   ├── components/
│   │   │   ├── SymptomForm.jsx
│   │   │   └── HistoryList.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
│
├── server/          # Node.js + Express Backend
│   ├── models/
│   │   └── Prediction.js
│   ├── index.js
│   └── package.json
│
├── model/           # Python ML Service
│   ├── app.py       # Flask API server
│   ├── train.py     # Model training script
│   ├── model.pkl    # Trained ML model
│   ├── symptoms.pkl # Symptoms list
│   └── requirements.txt
│
└── README.md
```

---

## 🚀 How To Run

You need **3 terminals** running simultaneously.

### Prerequisites
- Node.js installed
- Python 3.8+ installed
- MongoDB (optional — JSON fallback available)

---

### Terminal 1 — Python ML Model
```bash
cd model
pip install -r requirements.txt
python app.py
```
> Runs on: `http://localhost:5000`

---

### Terminal 2 — Node.js Backend
```bash
cd server
npm install
npm start
```
> Runs on: `http://localhost:5001`

---

### Terminal 3 — React Frontend
```bash
cd client
npm install
npm run dev
```
> Runs on: `http://localhost:5173`

---

## 🔄 How It Works

```
User selects symptoms (React UI)
         ↓
POST /api/predict → Node.js Server
         ↓
POST /model/predict → Python Flask
         ↓
Random Forest Model predicts disease
         ↓
Result returned → Saved in MongoDB
         ↓
User sees: Disease Name + Accuracy %
```

---

## 🤖 ML Model Details

- **Algorithm:** Random Forest Classifier
- **Trees:** 150 estimators
- **Training Data:** 1200 synthetic samples
- **Input:** 27 binary symptom features (0 = absent, 1 = present)
- **Output:** Disease name + confidence probability

---

## 📡 API Endpoints

### Python Flask (Port 5000)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/model/predict` | Predict disease from symptoms |
| GET | `/model/symptoms` | Get list of all symptoms |

### Node.js Express (Port 5001)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Get prediction & save to DB |
| GET | `/api/history` | Get prediction history |
| GET | `/api/symptoms` | Fetch available symptoms |
| DELETE | `/api/history/:id` | Delete a history record |

---

## 👨‍💻 Developer

Made with ❤️ as a college project.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
