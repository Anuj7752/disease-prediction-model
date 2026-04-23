const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const Prediction = require('./models/Prediction');

const app = express();
const PORT = process.env.PORT || 5001;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/disease_prediction';
mongoose.connect(MONGODB_URI)
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

// Local file fallback DB
const DB_FILE = path.join(__dirname, 'db.json');
if (!fs.existsSync(DB_FILE)) {
  fs.writeFileSync(DB_FILE, JSON.stringify([]));
}

// Routes
app.post('/api/predict', async (req, res) => {
  try {
    const { symptoms } = req.body;
    
    if (!symptoms || symptoms.length === 0) {
      return res.status(400).json({ error: 'Symptoms are required' });
    }

    // Call the Python ML Service
    const mlResponse = await axios.post('http://127.0.0.1:5000/model/predict', {
      symptoms
    });

    const { disease, accuracy } = mlResponse.data;

    // Save history to MongoDB
    const newPrediction = new Prediction({
      symptoms,
      disease,
      accuracy
    });
    
    let savedPrediction = null;
    try {
      await newPrediction.save();
      savedPrediction = newPrediction;
    } catch (dbError) {
      console.warn('Could not save to MongoDB. Using local JSON file DB.');
      const fallbackRecord = {
        _id: Date.now().toString(),
        symptoms,
        disease,
        accuracy,
        timestamp: new Date().toISOString()
      };
      
      const historyData = JSON.parse(fs.readFileSync(DB_FILE, 'utf-8'));
      historyData.unshift(fallbackRecord);
      fs.writeFileSync(DB_FILE, JSON.stringify(historyData, null, 2));

      savedPrediction = fallbackRecord;
    }

    res.json({
      disease,
      accuracy,
      savedPrediction: savedPrediction
    });

  } catch (error) {
    console.error('Prediction Error:', error.message);
    res.status(500).json({ error: 'Failed to predict disease. Ensure Model service is running.' });
  }
});

app.get('/api/history', async (req, res) => {
  try {
    // Check if MongoDB is connected (readyState 1 means connected)
    if (mongoose.connection.readyState !== 1) {
        const historyData = JSON.parse(fs.readFileSync(DB_FILE, 'utf-8'));
        return res.json(historyData);
    }
    const history = await Prediction.find().sort({ timestamp: -1 });
    res.json(history);
  } catch (error) {
    const historyData = JSON.parse(fs.readFileSync(DB_FILE, 'utf-8'));
    res.json(historyData); // Fallback to JSON file on error
  }
});

app.get('/api/symptoms', async (req, res) => {
    try {
        const mlResponse = await axios.get('http://127.0.0.1:5000/model/symptoms');
        res.json(mlResponse.data);
    } catch(err) {
        res.status(500).json({ error: 'Failed to fetch symptoms. Ensure Model service is running.' });
    }
});

app.delete('/api/history/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    // Check if MongoDB is connected (readyState 1 means connected)
    if (mongoose.connection.readyState === 1) {
        await Prediction.findByIdAndDelete(id);
    } else {
        let historyData = JSON.parse(fs.readFileSync(DB_FILE, 'utf-8'));
        historyData = historyData.filter(item => item._id !== id);
        fs.writeFileSync(DB_FILE, JSON.stringify(historyData, null, 2));
    }
    res.json({ message: 'Deleted successfully' });
  } catch (error) {
    console.error('Delete Error:', error.message);
    res.status(500).json({ error: 'Failed to delete record.' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
