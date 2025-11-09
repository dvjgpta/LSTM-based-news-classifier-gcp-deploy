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
git clone https://github.com/dvjgpta/LSTM-based-news-classifier-gcp-deploy.git
cd LSTM-based-news-classifier-gcp-deploy
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate
```
or conda:
```bash
conda create -n myenv
conda activate myenv
```
3.Install dependencies:
```bash
pip install -r requirements.txt
```
4. Ensure model.pt and vocab.pth are in the models/ directory.or you can train LSTM model using this [Link](https://github.com/dvjgpta/LSTM)
5 .Run the Flask app locally:
```bash
python app.py
```
###Docker Deployment
1. Make sure Docker is installed using:
```bash
pip install docker
```
2.Build the Docker image:
```bash
docker build -t news-lstm-app .
```
3. Run the container locally:
```bash
docker run -p 5000:5000 news-lstm-app
```


The project is hosted on Google Cloud platform and can be accessed using this [link](https://newsclassifier-app-1096413105645.us-central1.run.app)

