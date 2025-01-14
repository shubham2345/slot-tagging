**🧩 IOB Slot Tagging using NLP**


Kaggle Competition Link : https://www.kaggle.com/competitions/nlp-243-fall-24-hw-2-slot-tagging-of-utterances


This project focuses on slot tagging for natural language queries, aiming to extract meaningful information such as movies, directors, release years, and genres from user utterances. Using a dataset with IOB (Inside, Outside, Beginning) slot tags, various neural network models are developed and evaluated, including:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- CNN (Convolutional Neural Network)

The models leverage contextual embeddings such as GloVe to capture semantic and syntactic information. The study demonstrates the importance of sequential models and pre-trained embeddings in improving slot tagging accuracy, particularly in handling complex queries in dialogue systems.

📂 Dataset Structure
The dataset used for this project consists of the following columns:

ID: Unique identifier for each sample
Utterance: User query containing natural language text
IOB Tags: Corresponding slot tags in Inside, Outside, Beginning format

Example:

`ID: 1` 

`Utterance: "Show me movies directed by Christopher Nolan"` 

`IOB Tags: O O B-MOVIE O B-DIRECTOR I-DIRECTOR`

**🚀 Neural Network Models Used**
The project evaluates the following neural network architectures for slot tagging:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- CNN (Convolutional Neural Network)

These models are combined with GloVe embeddings to capture contextual semantics and improve tagging performance.

**🧪 Key Findings**
Sequential models (LSTM/GRU) outperform traditional models in handling complex queries.
Pre-trained embeddings (GloVe) significantly improve the accuracy of slot tagging.
The CNN model performs well but struggles with longer sequences compared to LSTM/GRU.


**📋 How to Run the Code**
Run the following command to execute the model on the dataset:


`python run.py data/hw2_train.csv data/hw2_test.csv data/submission.csv`