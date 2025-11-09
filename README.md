# News Classifier LSTM

A PyTorch-based **LSTM news classification model** deployed as a **Dockerized Flask app** on **Google Cloud Run** for real-time text predictions.

---

## Project Overview

This project implements a deep learning pipeline to classify news articles into four categories:  
- **World 🌍**  
- **Sports 🏅**  
- **Business 💼**  
- **Science/Tech 🔬**  

The LSTM model uses tokenized text inputs and predicts the category in real-time through a Flask web interface.

---

## Features

- **PyTorch LSTM** for sequence modeling and multi-class classification.  
- **Flask web app** with `/predict` route for text input and predictions.  
- **Dockerized deployment** for reproducibility and cloud hosting.  
- **Google Cloud Run** deployment for scalable, serverless inference.  
- Optimized memory usage for smooth deployment on Cloud Run.  

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker
- Google Cloud account (for Cloud Run deployment)

### Local Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/news-classifier-lstm.git
cd news-classifier-lstm
