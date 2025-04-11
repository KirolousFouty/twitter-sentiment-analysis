# **Twitter Sentiment Analysis**

## **Overview**
Twitter Sentiment Analysis is a machine learning-based application designed to analyze the sentiment of textual data, particularly tweets. The application predicts whether a given tweet expresses a **positive** or **negative** sentiment. Users can also provide feedback, allowing the model to improve over time.

## **Features**
- Sentiment classification using a trained **neural network**.
- Interactive desktop application with a **user-friendly interface**.
- Ability to **provide feedback**, which is used to improve the model.
- High-performance API built with **FastAPI**.
- Real-time text preprocessing and vectorization.

## **Technologies Used**
- **Python**: Main programming language.
- **PyQt5**: Used for building the **graphical user interface (GUI)**.
- **FastAPI**: High-performance web framework for backend services.
- **Pydantic**: Data validation and management.
- **scikit-learn**: Used for text vectorization and model training.
- **spaCy** & **NLTK**: Advanced natural language processing tools.
- **Uvicorn**: ASGI server for running the FastAPI backend.
- **Requests**: Library for handling HTTP requests.
- **numPy**: Used for matrix creation and operations.
- **sciPy**: Used for sparse matrices, Vstacks, and Hstacks.
- **Pickle**: Used for exporting the model weights.
- **Pandas**: Library for handling data.

## **System Architecture**
### **Backend (FastAPI)**
- **Endpoints:**
  - /predict/ → Accepts text input and returns the sentiment prediction.
  - /learn/ → Accepts text and user feedback to retrain the model.
- **Neural Network Model:**
  - Input Layer: Takes processed text features (max 280 tokens).
  - Hidden Layers: Three layers with **ReLU** activation.
  - Output Layer: **Sigmoid** function for binary classification.
  - **Optimizer:** Custom Adam optimizer for efficient learning.
  - **Batching:** Uses sparse matrices and batch size adjustments to optimize memory usage.

### **Frontend (PyQt5)**
- **Main Window:** User input for sentiment analysis.
- **Feedback Dialog:** Popup for user feedback.
- **Error Handling:** Input validation and HTTP request error handling.

## **How It Works**
1. User enters text (a tweet) and clicks **"Analyze Sentiment"**.
1. The app sends the text to the /predict/ API endpoint.
1. The model processes the text and returns a **sentiment prediction**.
1. The result is displayed in the app.
1. Users can provide feedback, which is sent to the /learn/ API to retrain the model.

## **Model Performance**
- Trained on **80% of dataset**, tested on **20%**.
- Achieved an accuracy of **~78%**.
- Uses **real-time text preprocessing** (removing noise, user mentions, etc.).


## **Future Improvements**
- Support for more sentiments (neutral, mixed).
- Deployment as a web-based application.
- Real-time Twitter API integration for live tweet analysis.

## **Authors**
- Kirolous Fouty, kirolous_fouty@aucegypt.edu
- Mohamed Sabry, momo12320@aucegypt.edu
