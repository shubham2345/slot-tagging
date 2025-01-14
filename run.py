import pandas as pd
import argparse
import torch
import torch.nn as nn
from model import EnhancedBiLSTMTaggerWithCNNLayers
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch.optim as optim
from seqeval.metrics import f1_score as seqeval_f1_score
from torch.optim.lr_scheduler import StepLR
from seqeval.scheme import IOB2
import matplotlib.pyplot as plt


random_state = 42

# Step 1: Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    sentences = df['utterances'].apply(lambda x: x.split()).to_list()
    tags = df['IOB Slot tags'].apply(lambda x: x.split()).to_list()
    return sentences, tags

# Step 2: Build Vocabulary
def build_vocab(sentences, tags):
    token_vocab = {"<PAD>": 0, "<UNK>": 1}
    tag_vocab = {"<PAD>": 0}
    for sentence in sentences:
        for token in sentence:
            if token not in token_vocab:
                token_vocab[token] = len(token_vocab)
    for tag_sequence in tags:
        for tag in tag_sequence:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)
    return token_vocab, tag_vocab

# Step 3: Convert to IDs
def convert_to_ids(sentences, tags, token_vocab, tag_vocab):
    token_ids = [
        torch.tensor([token_vocab.get(token, token_vocab['<UNK>']) for token in sentence])
        for sentence in sentences
    ]
    tag_ids = [
        torch.tensor([tag_vocab.get(tag, tag_vocab['<PAD>']) for tag in tag_sequence])
        for tag_sequence in tags
    ]
    return token_ids, tag_ids

# Step 4: Collate Function
def collate_fn(batch):
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] for item in batch]
    sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=token_vocab['<PAD>'])
    tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=tag_vocab['<PAD>'])
    return sentences_padded, tags_padded

# Step 5: Load GloVe Embeddings
def load_glove_embeddings(file_path, embedding_dim, token_vocab):
    print("Loading GloVe embeddings...")
    glove_embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
            glove_embeddings[word] = vector

    vocab_size = len(token_vocab)
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    for word, idx in token_vocab.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
        else:
            # embedding_matrix[idx] = torch.randn(embedding_dim)
            embedding_matrix[idx] = torch.zeros(embedding_dim)
    return embedding_matrix

# Step 6: Prediction Function
def predict_test_tags(model, test_token_ids, device):
    model.eval()
    all_predictions = []
    for token_ids in test_token_ids:
        token_ids = token_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(token_ids)
            pred_ids = outputs.argmax(dim=-1).squeeze().tolist()
            if isinstance(pred_ids, int):
                pred_ids = [pred_ids]
            pred_tags = [list(tag_vocab.keys())[tag_id] for tag_id in pred_ids]
            all_predictions.append(pred_tags)
    return all_predictions

# Step 7: Load Test Data
def load_test_data(file_path):
    df = pd.read_csv(file_path)
    sentences = df['utterances'].apply(lambda x: x.split()).tolist()
    ids = df['ID'].tolist()
    return ids, sentences

# Step 8: Main Function
def main():
    parser = argparse.ArgumentParser(description='Process training and test data.')
    parser.add_argument('training_data', type=str)
    parser.add_argument('test_data', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    train_sentences, train_tags = load_data(args.training_data)
    global token_vocab, tag_vocab
    token_vocab, tag_vocab = build_vocab(train_sentences, train_tags)
    train_token_ids, train_tag_ids = convert_to_ids(train_sentences, train_tags, token_vocab, tag_vocab)

    train_data, val_data = train_test_split(list(zip(train_token_ids, train_tag_ids)), test_size=0.1, random_state=random_state, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=15, shuffle=False, collate_fn=collate_fn)

    # Hyperparameters
    EMBEDDING_DIM = 768
    HIDDEN_DIM = 512
    LEARNING_RATE = 0.003
    NUM_EPOCHS = 50
    DROPOUT_RATE = 0.3
    PATIENCE = 5

    model = EnhancedBiLSTMTaggerWithCNNLayers(len(token_vocab), len(tag_vocab), EMBEDDING_DIM, HIDDEN_DIM, token_vocab, dropout=DROPOUT_RATE)
    model.embedding = nn.Embedding(len(token_vocab), EMBEDDING_DIM, padding_idx=token_vocab['<PAD>'])

    loss_fn = nn.CrossEntropyLoss(ignore_index=tag_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses = []
    val_losses = []
    seqeval_f1_scores = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0

        # Training Loop
        for token_ids, tag_ids in train_loader:
            token_ids, tag_ids = token_ids.to(device), tag_ids.to(device)

            optimizer.zero_grad()
            outputs = model(token_ids)
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        scheduler.step()

        # Validation Loop
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_tags = []

        with torch.no_grad():
            for token_ids, tag_ids in val_loader:
                token_ids, tag_ids = token_ids.to(device), tag_ids.to(device)
                outputs = model(token_ids)
                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
                total_val_loss += loss.item()

                predictions = outputs.argmax(dim=-1)

                for i in range(predictions.size(0)):
                    mask = tag_ids[i] != tag_vocab['<PAD>']
                    filtered_pred = predictions[i][mask].tolist()
                    filtered_tag = tag_ids[i][mask].tolist()

                    all_predictions.append(filtered_pred)
                    all_tags.append(filtered_tag)

        all_predictions_iob = [[list(tag_vocab.keys())[tag] for tag in seq] for seq in all_predictions]
        all_tags_iob = [[list(tag_vocab.keys())[tag] for tag in seq] for seq in all_tags]

        val_f1 = seqeval_f1_score(all_tags_iob, all_predictions_iob, mode='strict', scheme=IOB2)

        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Seqeval F1 Score: {val_f1:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        seqeval_f1_scores.append(val_f1)

        # Early Stopping Check
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        #     print("Validation loss improved. Resetting patience counter.")
        # else:
        #     patience_counter += 1
        #     print(f"No improvement in validation loss. Patience counter: {patience_counter}/{PATIENCE}")

        # if patience_counter >= PATIENCE:
        #     print("Early stopping triggered. Stopping training.")
        #     break

    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", color='blue', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color='red', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Seqeval F1 Score
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(seqeval_f1_scores) + 1), seqeval_f1_scores, label="Seqeval F1 Score", color='green', marker='x')
    plt.xlabel("Epochst")
    plt.ylabel("F1 Score")
    plt.title("Seqeval F1 Score Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    # Test Predictions
    test_ids, test_sentences = load_test_data(args.test_data)
    test_token_ids = convert_to_ids(test_sentences, [[] for _ in test_sentences], token_vocab, tag_vocab)[0]
    test_predictions = predict_test_tags(model, test_token_ids, device)

    # Prepare Submission File
    submission_data = [[test_ids[i], ' '.join(test_predictions[i])] for i in range(len(test_ids))]
    submission_df = pd.DataFrame(submission_data, columns=['ID', 'IOB Slot tags'])
    submission_df.to_csv(args.output, index=False)
    print("Submission file generated successfully.")


if __name__ == "__main__":
    main()

