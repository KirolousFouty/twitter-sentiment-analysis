import sys
import subprocess
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QDesktopWidget, QMenuBar, QAction, QMessageBox, QDialog
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import contractions
import spacy
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, issparse, vstack
from scipy.sparse import hstack
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')
stopword_list = set(stopwords.words('english'))
tokenizer = word_tokenize
#df=pd.read_csv('cleaned_data2.csv')
vectorizer = CountVectorizer()
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define preprocessing functions
def lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def remove_accented_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def expand_contractions(text):
    text = contractions.fix(text)
    return re.sub("'", "", text)

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def remove_stopwords(text):
    tokens = tokenizer(text)
    return ' '.join([token for token in tokens if token not in stopword_list])

def remove_mentions(text):
    return re.sub(r'@[A-Za-z0-9]+', '', text)

def remove_url(text):
    return re.sub(r'https?://[A-Za-z0-9./]+', '', text)

def remove_hashtag(text):
    return re.sub(r'[^a-zA-Z\s]', ' ', text)

def remove_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)
def normalize_corpus(doc):
    doc = remove_accented_chars(doc)
    doc = expand_contractions(doc)
    doc = doc.lower()
    doc = remove_mentions(doc)
    doc = remove_url(doc)
    doc = remove_hashtag(doc)
    doc = remove_repeated_chars(doc)
    doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
    doc = remove_special_characters(doc)
    doc = re.sub(' +', ' ', doc)
    doc = remove_stopwords(doc)
    doc = lemmatize(doc)
    return doc
#NNNNNNN

def normalize_mentions(num):
    return float(((num-0)/3887-0))

def relu(Z):
    Z.data = np.maximum(0, Z.data)
    return Z

def derivative_relu(Z):
    if issparse(Z):
        # Handle sparse matrix: apply derivative on non-zero elements
        Z_data = np.asarray(Z.data)  # Convert memoryview to numpy array
        derivative_data = (Z_data > 0).astype(float)  # Apply the derivative to the numpy array
        # Return as sparse matrix in CSR format
        return csr_matrix((derivative_data, Z.indices, Z.indptr), shape=Z.shape)
    else:
        # Handle dense matrix: directly apply the derivative and convert to sparse
        derivative_data = (Z > 0).astype(float)  # Apply the derivative to the numpy array
        # Convert the result to sparse (CSR format)
        return csr_matrix(derivative_data)
def sigmoid(Z):
    Z_data = np.asarray(Z.data)  # Convert memoryview to numpy array
    Z.data = 1 / (1 + np.exp(-Z_data))  # Apply sigmoid to the numpy array
    return Z

def derivative_sigmoid(A):
    return A.multiply(1 - A)  # Sigmoid's derivative using the output (A)

class Layer:
    def __init__(self, input_dim, output_dim, activation="relu"):
        self.W = csr_matrix(
            np.random.standard_normal(size=(input_dim, output_dim)) * np.sqrt(2 / input_dim)
        )
        self.b = csr_matrix((1, output_dim))
        self.activation_name = activation
        self.A_prev = None
        self.Z = None
        self.A = None

        # Define activation functions dynamically
        if activation == "relu":
            self.activation = relu
            self.activation_derivative = derivative_relu
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = derivative_sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, A_prev):
        """
        Perform forward propagation for this layer.
        """
        #self.A_prev = A_prev
        self.A_prev = csr_matrix(A_prev) if not issparse(A_prev) else A_prev
        #print(f"A_prev shape: {self.A_prev.shape}, b shape: {self.b.shape}, W shape: {self.W.shape}")
        bias_sparse = csr_matrix(np.ones((self.A_prev.shape[0], 1))) @ csr_matrix(self.b)  # Efficient sparse broadcast  # Convert bias to sparse format
        #print(f"A_prev shape: {self.A_prev.shape}, bias_sparse shape: {bias_sparse.shape}, W shape: {self.W.shape}")
        self.Z = self.A_prev.dot(self.W) + bias_sparse  # Bias is broadcasted efficiently
        #print(self.Z)
        #print(f"Forward - Z shape: {self.Z.shape}, A_prev shape: {self.A_prev.shape}, b shape: {self.b.shape}, W shape: {self.W.shape}")
        # Apply the activation function
        self.A = self.activation(self.Z)
        #print(f"Forward - Z shape: {self.Z.shape}, A shape: {self.A.shape}, b shape: {self.b.shape}, W shape: {self.W.shape}")
        return self.A

    def backward(self, dA, m, is_output_layer=False, y=None):
        """
        Perform backward propagation for this layer.
        """
        if is_output_layer:
            #print('OUTPUT')
            y = y.reshape(-1, 1)
            #print(y)
            dZ = self.A - y  # For output layer, dZ is the difference between predicted and true values
        else:
            # Ensure dA and activation derivative have the same shape
            if isinstance(dA, csr_matrix) and isinstance(self.activation_derivative(self.Z), csr_matrix):
                dZ = dA.multiply(self.activation_derivative(self.Z))  # Sparse element-wise multiplication
            else:
                dZ = dA * self.activation_derivative(self.Z)  # For dense matrices (if any)
        dW = self.A_prev.T.dot(dZ) / m  # Sparse matrix multiplication
        dW = dW.tocsr()
        t=csr_matrix(dZ)
        db = t.sum(axis=0)
        db=db/m
        db = csr_matrix(db).reshape(1, -1)
        #db = csr_matrix(dZ).sum(axis=0) / m

        dA_prev = dZ.dot(self.W.T)  # Sparse matrix multiplication

        return dA_prev, dW, db

class OptimizedNeuralNetworkClassifier:
    def __init__(self, layers_config, learning_rate=0.01, batch_size=5, optimizer="adam", beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        layers_config: List of dictionaries with keys {'input_dim', 'output_dim', 'activation'}
        """
        self.layers = [Layer(**layer) for layer in layers_config]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.opt_cache = self._initialize_optimizer_cache()

    def _initialize_optimizer_cache(self):
        cache = {"m": [], "v": [], "t": 0}
        for layer in self.layers:
            # Initialize m and v as sparse matrices with the same shape as the layer's weights (W) and biases (b)
            m_dW = csr_matrix(layer.W.shape)
            m_db = csr_matrix(layer.b.shape)
            v_dW = csr_matrix(layer.W.shape)
            v_db = csr_matrix(layer.b.shape)
            
            cache["m"].append({"dW": m_dW, "db": m_db})
            cache["v"].append({"dW": v_dW, "db": v_db})
        
        return cache

    def _forward(self, X):
        """
        Forward pass through all layers.
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def _backward(self, y, A):
        """
        Backward pass through all layers.
        """
        m = y.shape[0]  # Number of samples in the batch
        grads = []
        dA = None  # Gradient of the loss with respect to activations

        # Ensure 'y' is a column vector (2D array)
        if len(y.shape) == 1:  # If 'y' is a 1D array (series)
            y = y.reshape(-1, 1)
        y_sparse = csr_matrix(y)

        # Loop backward through the layers
        for i in reversed(range(len(self.layers))):
            is_output_layer = (i == len(self.layers) - 1)
            layer = self.layers[i]

            # Debug print statements to track shapes
            #print(f"Layer {i}:")
            #print(f"    y shape: {y.shape}")
            #print(f"    A shape: {A.shape}")
            #if dA is not None:
                #print(f"    dA shape (from next layer): {dA.shape}")

            # Check dimensions before calling backward
            #if not is_output_layer and dA.shape[1] != layer.W.shape[0]:
                #print(f"    WARNING: dA shape {dA.shape} does not match expected input shape {layer.W.shape}.")
                #print(f"dA type {dA} and W type {layer.W}.")
                #raise ValueError("Mismatch between gradient and layer input dimensions.")

            try:
                # For output layer, compute loss gradient (A - y); otherwise, use dA from the next layer
                x=A - y_sparse
                #print(x)
                #print(x.type)
                dA, dW, db = layer.backward(
                    dA if not is_output_layer else x,
                    m,
                    is_output_layer,
                    y_sparse
                )
                grads.insert(0, {"dW": dW, "db": db})  # Add current layer's gradients to the list
            except ValueError as e:
                print(f"    Error during backward pass in Layer {i}: {e}")
                raise

        return grads

    def _update_parameters(self, grads):
        self.opt_cache["t"] += 1
        t = self.opt_cache["t"]

        for i, layer in enumerate(self.layers):
            if self.optimizer == "adam":
                # Update first moment estimate (m)
                self.opt_cache["m"][i]["dW"] = self.beta1 * self.opt_cache["m"][i]["dW"] + (1 - self.beta1) * grads[i]["dW"]
                self.opt_cache["m"][i]["db"] = self.beta1 * self.opt_cache["m"][i]["db"] + (1 - self.beta1) * grads[i]["db"]

                #print(grads[i]["dW"])
                #print(grads[i]["dW"].type)
                #print(grads[i].type)
                #print(grads.type)
                #print(isinstance(grads[i]["dW"], csr_matrix))
                # For sparse gradients, use .multiply() to square the values
                if isinstance(grads[i]["dW"], csr_matrix):
                    self.opt_cache["v"][i]["dW"] = self.beta2 * self.opt_cache["v"][i]["dW"] + (1 - self.beta2) * grads[i]["dW"].multiply(grads[i]["dW"])
                    self.opt_cache["v"][i]["db"] = self.beta2 * self.opt_cache["v"][i]["db"] + (1 - self.beta2) * grads[i]["db"].multiply(grads[i]["db"])
                else:
                    # This case shouldn't happen because you specified the need for sparse matrices
                    raise ValueError("Gradients should be sparse matrices.")

                # Correct for bias in moment estimates
                m_hat_dW = self.opt_cache["m"][i]["dW"] / (1 - self.beta1 ** t)
                m_hat_db = self.opt_cache["m"][i]["db"] / (1 - self.beta1 ** t)

                v_hat_dW = self.opt_cache["v"][i]["dW"] / (1 - self.beta2 ** t)
                v_hat_db = self.opt_cache["v"][i]["db"] / (1 - self.beta2 ** t)

                v_hat_dW = csr_matrix(v_hat_dW) if not isinstance(v_hat_dW, csr_matrix) else v_hat_dW

                # Apply the square root to the sparse matrix
                sqrt_v_hat_dW = csr_matrix(v_hat_dW).power(0.5)  # Sparse square root
                sqrt_v_hat_db = csr_matrix(v_hat_db).power(0.5)  # Sparse square root for db

                # Add epsilon to the sparse matrix (make sure it's sparse)
                epsilon_sparse_dW = csr_matrix(np.ones(sqrt_v_hat_dW.shape)) * self.epsilon
                epsilon_sparse_db = csr_matrix(np.ones(sqrt_v_hat_db.shape)) * self.epsilon

                sqrt_v_hat_dW_with_epsilon = sqrt_v_hat_dW + epsilon_sparse_dW
                sqrt_v_hat_db_with_epsilon = sqrt_v_hat_db + epsilon_sparse_db
                # Update weights and biases using sparse operations
                layer.W -= self.learning_rate * m_hat_dW / (sqrt_v_hat_dW_with_epsilon)
                layer.b -= self.learning_rate * m_hat_db / (sqrt_v_hat_db_with_epsilon)

                # Ensure that the weights (W) and biases (b) are still sparse matrices after the update
                layer.W = csr_matrix(layer.W)  # Make sure W is sparse
                layer.b = csr_matrix(layer.b)  # Make sure b is sparse

            else:
                # Non-Adam optimizer: Use sparse updates (this case should not happen for your setup)
                layer.W -= self.learning_rate * grads[i]["dW"]
                layer.b -= self.learning_rate * grads[i]["db"]

                # Ensure that the weights (W) and biases (b) are sparse matrices after the update
                layer.W = csr_matrix(layer.W)  # Make sure W is sparse
                layer.b = csr_matrix(layer.b)  # Make sure b is sparse

    def fit(self, X, y, epochs=10):
        ecnt=0
        patience = 5
        best_loss = float('inf')
        for epoch in range(epochs):
            bcnt=0
            epoch_loss = 0  # To accumulate the total loss for the epoch
            epoch_accuracy = 0  # To accumulate accuracy for the epoch
            for batch_X, batch_y in self._get_mini_batches(X, y):
                A = self._forward(batch_X)
                #print(batch_y.type)
                grads = self._backward(batch_y, A)
                #print(grads)
                #grads is a list
                self._update_parameters(grads)
                bcnt=bcnt+1
                batch_loss = self._compute_loss(batch_X, batch_y)  # Compute loss for the batch
                batch_accuracy = self._compute_accuracy(batch_X, batch_y)  # Compute accuracy for the batch
                # Accumulate batch-wise loss and accuracy
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                print(f'BATCH: {bcnt}, Batch Loss: {batch_loss:.4f}, Batch Accuracy: {batch_accuracy:.4f}')

            avg_epoch_loss = epoch_loss / bcnt
            avg_epoch_accuracy = epoch_accuracy / bcnt

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience = 5  # Reset patience if improvement is made
            else:
                patience -= 1
                if patience == 0:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_accuracy:.4f}")
            ecnt += 1
            #print(f'EPOCH: {ecnt}')

    def _compute_loss(self, X, y):
        A = self._forward(X)  # Forward pass, A will be a sparse matrix
        # Ensure y and A are sparse if they aren't already
        #y = csr_matrix(y) if not isinstance(y, csr_matrix) else y
        #A = csr_matrix(A) if not isinstance(A, csr_matrix) else A
        #if isinstance(y, pd.Series):
        y = csr_matrix(y.reshape(-1, 1))  # Convert to sparse column vector
            #y = y.values.reshape(-1, 1) # Convert to sparse column vector
        A_clipped = A.copy()
        A_clipped.data = np.clip(A_clipped.data, 1e-8, 1 - 1e-8)

        # Element-wise log on the sparse data
        log_A = A_clipped.copy()
        log_A.data = np.log(log_A.data)

        log_1_minus_A = A_clipped.copy()
        log_1_minus_A.data = np.log(1 -  np.array(A_clipped.data))

        # Calculate loss using sparse matrix operations
        # y.multiply(log_A):  y * log(A)
        # (1 - y).multiply(log_1_minus_A):  (1 - y) * log(1 - A)
        one_minus_y = csr_matrix((1 - y.toarray()))  # Convert (1 - y) into a sparse matrix
        #print('Shapes: log_A: ', log_A.shape, ' log_1_minus_A: ',log_1_minus_A.shape,' one_minus_y: ',one_minus_y.shape)
        loss_matrix = y.multiply(log_A) + one_minus_y.multiply(log_1_minus_A)
        
        #one_minus_y = csr_matrix((1 - y.toarray()))  # Convert (1 - y) into a sparse matrix
        #loss_matrix = y.multiply(log_A) + one_minus_y.multiply(log_1_minus_A)
        #loss_matrix = y.multiply(log_A) + (1 - y).multiply(log_1_minus_A)
        # Compute the element-wise loss
        #loss_matrix = (y.multiply(np.log(A + 1e-8)) + (1 - y).multiply(np.log(1 - A + 1e-8)))
        
        # Return the sum of the loss (as a sparse matrix)
        #loss = loss_matrix.sum() / X.shape[0]
        return -loss_matrix.sum() / X.shape[0]

    def _compute_accuracy(self, X, y):
        # Get predictions from the model
        predictions = self.predict(X)
        
        # If predictions are sparse, convert to dense before comparison, or handle sparse efficiently
        if isinstance(predictions, csr_matrix):
            predictions = predictions.toarray().flatten()  # Convert sparse to dense for comparison
        
        if isinstance(y, csr_matrix):
            y = y.toarray().flatten()  # Convert sparse labels to dense if necessary

        # Compute accuracy as the mean of correct predictions
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def _compute_accuracy2(self, A, y):
        #predictions = self.predict(X)
        return np.mean(A == y)

    def predict(self, X):
        A = self._forward(X)
        return (A > 0.5).astype(int)
    def predict_2(self, X):
        A = self._forward(X)
        return (A)

    def _get_mini_batches(self, X, y):
        m = X.shape[0]

        if m % self.batch_size != 0:
            padding_size = self.batch_size - (m % self.batch_size)
        else:
            padding_size = 0
        #print(padding_size)
        if padding_size > 0:
            padding_X = csr_matrix(np.zeros((padding_size, X.shape[1])))
            X_padded = vstack([X, padding_X])
            y_padded = np.pad(y, ((0, padding_size),), mode='constant', constant_values=0)
        else:
            X_padded = X
            y_padded = y
        
        assert X_padded.shape[0] == y_padded.shape[0], "X and y must have the same number of samples after padding."
        
        indices = np.arange(X_padded.shape[0])
        np.random.shuffle(indices)
        for i in range(0,  X_padded.shape[0], self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_X = X_padded[batch_indices]  # No need to convert to dense if it's sparse
            # Extract batch_y
            if isinstance(y_padded, pd.Series):
                batch_y = y_padded.iloc[batch_indices].values  # Use .iloc for Pandas Series
            else:
                batch_y = y_padded[batch_indices]  # Works for NumPy arrays
            # Convert batch_y to numpy array if it is a Series
            #batch_y = batch_y.values if isinstance(batch_y, pd.Series) else batch_y
            yield batch_X, batch_y

    def get_weights(self):
        """
        Extract weights and biases from all layers of the model.
        Returns:
            weights: A list of dictionaries containing 'W' (weights) and 'b' (biases)
                    for each layer.
        """
        weights = []
        for i, layer in enumerate(self.layers):
            layer_weights = {
                "W": layer.W,  # Sparse weight matrix
                "b": layer.b,  # Sparse bias matrix
            }
            weights.append(layer_weights)
        return weights

    # Save weights and biases to a .pkl file
    def save_weights_to_pkl(model, filename):
        weights = model.get_weights()  # Get weights and biases
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)

    # Load weights and biases from a .pkl file
    def load_weights_from_pkl(model, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        
        # Assign weights and biases back to the model's layers
        for layer, layer_weights in zip(model.layers, weights):
            layer.W = layer_weights["W"]
            layer.b = layer_weights["b"]
        print("Model weights loaded from", filename)

layers_config = [
    {"input_dim": 242935, "output_dim": 256, "activation": "relu"},
    {"input_dim": 256, "output_dim": 128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 1, "activation": "sigmoid"},
]

nn = OptimizedNeuralNetworkClassifier(layers_config, learning_rate=0.01, batch_size=5, optimizer="adam")
nn.load_weights_from_pkl('model_weights.pkl')

learn_list=[]

app = FastAPI()

class InputData_Predict(BaseModel):
    text: str
    numeric_value: float  # Accept numeric input


class InputData_Train(BaseModel):
    text: str
    numeric_value: float  # Accept numeric input
    label: int

@app.post("/predict/")
async def predict(input_data: InputData_Predict):
    # Preprocess the text input
    normalized_text = normalize_corpus(input_data.text)
    #vectorizer = CountVectorizer()
    # Convert normalized text into features (assuming binary encoding)
    text_features = vectorizer.transform([normalized_text])
    #text_features += [0] * (256 - len(text_features))  # Pad to match text feature size
    mentions=normalize_mentions(input_data.numeric_value)
    # Combine text features with numeric value
    #mentions=0
    #mentions = np.array(input_data.numeric_value).reshape(1, -1)
    mentions = np.array(0).reshape(1, -1)
    mentions = csr_matrix(mentions)
    #mentions=mentions.reshape(-1, 1)
    count_vect_hstack = hstack([text_features, mentions]) 
    # Generate prediction
    prediction = nn.predict_2(count_vect_hstack)
    
    return {"prediction": prediction.data[0]}
@app.post("/learn/")
async def learn(input_data: InputData_Train):
    # Preprocess the text input
    normalized_text = normalize_corpus(input_data.text)
    #vectorizer = CountVectorizer()
    # Convert normalized text into features (assuming binary encoding)
    text_features = vectorizer.transform([normalized_text])
    #text_features += [0] * (256 - len(text_features))  # Pad to match text feature size
    mentions=normalize_mentions(input_data.numeric_value)
    # Combine text features with numeric value
    mentions = np.array(input_data.numeric_value).reshape(1, -1)
    mentions = csr_matrix(mentions)
    #mentions=mentions.reshape(-1, 1)
    count_vect_hstack = hstack([text_features, mentions]) 
    label=input_data.label
    # Generate prediction
    #learn_list.append(count_vect_hstack,label)
    label = np.array(0).reshape(1, -1)
    label = csr_matrix(label)
    llx=hstack([count_vect_hstack, label])
    learn_list.append(llx)
    if(len(learn_list)>=5):
        input=csr_matrix((0, 242934))
        input_labels=[]
        for i in learn_list:
            extracted_xy = i[:, :242934]  # Columns corresponding to xy
            extracted_l = i[:, 242934:242935]  # Columns corresponding to l
            #extracted_l_dense = extracted_l.toarray()
            input = vstack([input, extracted_xy])
            input_labels.append(extracted_l.data)
        nn.fit(input,input_labels)
        learn_list.clear()
        # Show a popup window with the sentence
        # msg_box = QMessageBox()
        # msg_box.setIcon(QMessageBox.Information)
        # msg_box.setText(f"Model has been fitted on the new data.")
        # msg_box.setWindowTitle("Information")
        # msg_box.setStandardButtons(QMessageBox.Ok)
        # msg_box.exec_()
        print(f"Model Has Fitted on The New Data")
        return {f"Model Has Fitted on The New Data"}
    else:
        # Show a popup window with the sentence
        # msg_box = QMessageBox()
        # msg_box.setIcon(QMessageBox.Information)
        # msg_box.setText(f"Model Appended The Learning Entry to It's Learning List, Refitting After {5-len(learn_list)} More Entries")
        # msg_box.setWindowTitle("Information")
        # msg_box.setStandardButtons(QMessageBox.Ok)
        # msg_box.exec_()
        print(f"Model Appended The Learning Entry to It's Learning List, Refitting After {5-len(learn_list)} More Entries")
        return {f"Model Appended The Learning Entry to It's Learning List, Refitting After {5-len(learn_list)} More Entries"}
    #prediction = nn.predict_2(count_vect_hstack)



class SentimentAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentiment Analysis Tool")
        self.setGeometry(100, 100, 600, 400)

        # Center the window
        self.center_window()

        # Start the backend process in parallel
        self.start_backend()

        # Initialize UI components
        self.initUI()

    def center_window(self):
        """Center the window on the screen."""
        frame_geometry = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

    def start_backend(self):
        """Start the backend script in parallel."""
        self.backend_process = subprocess.Popen(
            ["python", "Backend.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def initUI(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout for central widget
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Menu bar
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # Add menu items
        file_menu = menu_bar.addMenu("File")
        help_menu = menu_bar.addMenu("Help")

        # File menu actions
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu actions
        help_action = QAction("Documentation", self)
        help_action.triggered.connect(self.show_help)
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(help_action)
        help_menu.addAction(about_action)

        # Title Label
        title_label = QLabel("Welcome to Sentiment Analysis Tool")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Input Label and Text Input
        input_layout = QVBoxLayout()
        self.input_label = QLabel("Enter your text:")
        self.input_label.setFont(QFont("Arial", 14))
        input_layout.addWidget(self.input_label)

        self.text_input = QLineEdit()
        self.text_input.setFont(QFont("Arial", 12))
        input_layout.addWidget(self.text_input)
        main_layout.addLayout(input_layout)

        # Buttons
        button_layout = QHBoxLayout()

        self.analyze_button = QPushButton("Analyze Sentiment")
        self.analyze_button.setFont(QFont("Arial", 12))
        self.analyze_button.clicked.connect(self.analyze_sentiment)
        button_layout.addWidget(self.analyze_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFont(QFont("Arial", 12))
        self.reset_button.clicked.connect(self.reset_input)
        button_layout.addWidget(self.reset_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.setFont(QFont("Arial", 12))
        self.exit_button.clicked.connect(self.close)
        button_layout.addWidget(self.exit_button)

        main_layout.addLayout(button_layout)

        # Result Label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

    def predict(self, text):
        """Send POST request to the predict endpoint."""
        url = "http://127.0.0.1:8000/predict/"
        try:
            response = requests.post(url, json={"text": text, "numeric_value": 0.0})  # Include numeric_value
            response.raise_for_status()
            return response.json().get("prediction", "Error")
        except requests.RequestException as e:
            QMessageBox.critical(self, "Error", f"Prediction request failed: {e}")
            return "Error"

    def learn(self, text, feedback):
        """Send POST request to the learn endpoint."""
        url = "http://127.0.0.1:8000/learn/"
        try:
            response = requests.post(url, json={"text": text, "numeric_value": 0.0, "label": feedback})
            response.raise_for_status()
        except requests.RequestException as e:
            QMessageBox.critical(self, "Error", f"Feedback request failed: {e}")

    def analyze_sentiment(self):
        """Process the input text, display sentiment, and collect feedback."""
        text = self.text_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text to analyze.")
            return
        if len(text) > 280:
            QMessageBox.warning(self, "Input Error", "Text exceeds Twitter's character limit of 280 characters.")
            return
        
        sentiment = self.predict(text)
        # sentiment = round(sentiment, 5)
        sentimentText = "Positive" if sentiment > 0.5 else "Negative"
        self.result_label.setText(f"Sentiment: {sentimentText} ({sentiment})")

        # Show feedback popup
        self.show_feedback_popup(text)

    def show_feedback_popup(self, text):
        """Show a feedback popup dialog."""
        dialog = FeedbackDialog(text, self)
        dialog.exec_()

    def reset_input(self):
        """Reset the input field and result label."""
        self.text_input.clear()
        self.result_label.clear()

    def show_help(self):
        """Show documentation or help message."""
        QMessageBox.information(
            self, "Documentation",
            "This tool analyzes the sentiment of the input text.\n\n"
            "1. Enter your text in the text box.\n"
            "2. Click 'Analyze Sentiment' to process.\n"
            "3. Provide feedback on the prediction.\n"
            "4. Reset clears the input."
        )

    def show_about(self):
        """Show about message."""
        QMessageBox.about(
            self, "About",
            "Sentiment Analysis Tool\n"
            "Version 1.0\n\n"
            "Machine Learning Project\n"
            "Developed by Kirolous Fouty and Mohamed Sabry.\n"
            "Contact: kirolous_fouty@aucegypt.edu , momo12320@aucegypt.edu"
        )

    def closeEvent(self, event):
        """Handle the window close event to terminate the backend process."""
        if hasattr(self, "backend_process"):
            self.backend_process.terminate()
        super().closeEvent(event)


class FeedbackDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.setWindowTitle("Feedback")
        self.setGeometry(200, 200, 300, 200)
        self.initUI()

        # Center the dialog window
        frame_geometry = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

    def initUI(self):
        layout = QVBoxLayout()

        label = QLabel("Feedback: what was the sentiment?")
        label.setFont(QFont("Arial", 14))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        button_layout = QHBoxLayout()

        positive_button = QPushButton("Positive")
        positive_button.clicked.connect(self.positive_feedback)
        button_layout.addWidget(positive_button)

        negative_button = QPushButton("Negative")
        negative_button.clicked.connect(self.negative_feedback)
        button_layout.addWidget(negative_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def positive_feedback(self):
        self.parent().learn(self.text, 1)
        self.accept()

    def negative_feedback(self):
        self.parent().learn(self.text, 0)
        self.accept()

import threading
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    import uvicorn

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Set modern style
    window = SentimentAnalysisApp()
    window.show()
    sys.exit(app.exec_())
