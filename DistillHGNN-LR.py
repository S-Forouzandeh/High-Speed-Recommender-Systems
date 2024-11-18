
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import time


def load_interactions_and_count(graph_path):
    interactions_df = pd.read_csv(
        graph_path,
        dtype={'user_id': str, 'item_id': str, 'review': str},
        low_memory=False
    )

    print(interactions_df.head())

    G = nx.Graph()
    edges = [(row['user_id'], row['item_id']) for _, row in interactions_df.iterrows()]
    G.add_edges_from(edges)

    # Create an adjacency matrix as a numpy array
    adj_matrix = nx.to_numpy_array(G)

    # Convert adjacency matrix to PyTorch tensor
    adj_matrix_tensor = torch.FloatTensor(adj_matrix)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    return G, num_nodes, num_edges, interactions_df, adj_matrix_tensor

def pad_adjacency_matrices(adj_matrix1, adj_matrix2):
    size1 = adj_matrix1.shape[0]
    size2 = adj_matrix2.shape[0]
    
    new_size = max(size1, size2)
    
    # Ensure matrices are torch tensors
    padded_adj_matrix1 = torch.zeros((new_size, new_size), dtype=adj_matrix1.dtype)
    padded_adj_matrix2 = torch.zeros((new_size, new_size), dtype=adj_matrix2.dtype)
    
    # Copy the original matrices
    padded_adj_matrix1[:size1, :size1] = adj_matrix1
    padded_adj_matrix2[:size2, :size2] = adj_matrix2

    return padded_adj_matrix1, padded_adj_matrix2

def create_homogeneous_adjacency_matrix(interactions_df):
    # Extract unique user and item IDs
    unique_users = interactions_df['user_id'].unique()
    unique_items = interactions_df['item_id'].unique()

    # Map user and item IDs to a single list
    all_nodes = np.concatenate([unique_users, unique_items])
    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}

    num_nodes = len(all_nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Fill the adjacency matrix based on interactions
    for _, row in interactions_df.iterrows():
        user_idx = node_to_index[row['user_id']]
        item_idx = node_to_index[row['item_id']]
        adj_matrix[user_idx, item_idx] = 1
        adj_matrix[item_idx, user_idx] = 1  # Optional, depending on how you want to represent interactions

    return torch.tensor(adj_matrix, dtype=torch.float32), node_to_index

def create_hypergraphs_and_incidence_matrices(interactions_df):
    # Get unique users and items
    users = interactions_df['user_id'].unique()
    items = interactions_df['item_id'].unique()

    num_users = len(users)
    num_items = len(items)

    # Mapping from user/item IDs to indices
    user_to_index = {user: idx for idx, user in enumerate(users)}
    item_to_index = {item: idx for idx, item in enumerate(items)}

    # Initialize incidence matrices for two hypergraphs
    user_hypergraph_incidence = np.zeros((num_users, num_items))
    item_hypergraph_incidence = np.zeros((num_items, num_users))

    # Fill the incidence matrices based on interactions
    for _, row in interactions_df.iterrows():
        user_idx = user_to_index[row['user_id']]
        item_idx = item_to_index[row['item_id']]
        
        # Fill user-item incidence matrix
        user_hypergraph_incidence[user_idx, item_idx] = 1
        
        # Fill item-user incidence matrix
        item_hypergraph_incidence[item_idx, user_idx] = 1

    # Convert incidence matrices to PyTorch tensors
    user_hypergraph_tensor = torch.tensor(user_hypergraph_incidence, dtype=torch.float32)
    item_hypergraph_tensor = torch.tensor(item_hypergraph_incidence, dtype=torch.float32)

    # Return the two incidence matrices and the mapping dictionaries
    return user_hypergraph_tensor, item_hypergraph_tensor, user_to_index, item_to_index

def pad_matrix(matrix, target_rows, target_cols):
    current_rows, current_cols = matrix.shape
    row_padding = target_rows - current_rows
    col_padding = target_cols - current_cols
    if row_padding > 0 or col_padding > 0:
        # If using torch:
        matrix = F.pad(matrix, (0, col_padding, 0, row_padding), mode='constant', value=0)
    return matrix

def compute_degree_matrices(incidence_matrix):
    # Ensure incidence_matrix is a torch tensor
    if isinstance(incidence_matrix, np.ndarray):
        incidence_matrix = torch.tensor(incidence_matrix, dtype=torch.float32)
    
    D_v = torch.diag(torch.sum(incidence_matrix, dim=1))
    D_e = torch.diag(torch.sum(incidence_matrix, dim=0))

    # Regularize by adding a small value to diagonal elements to avoid division by zero
    D_v_inv_sqrt = torch.inverse(torch.sqrt(D_v + torch.eye(D_v.size(0)) * 1e-10))
    D_e_inv = torch.inverse(D_e + torch.eye(D_e.size(0)) * 1e-10)

    return D_v_inv_sqrt, D_e_inv

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, 1)  # Final layer for binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Apply sigmoid activation for binary output
        return x

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            predicted = (logits > 0.5).float()  # Convert logits to binary predictions
        return predicted

# Define HypergraphNN model
class HypergraphNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HypergraphNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.theta = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.theta_final = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.theta_final)

    def forward(self, incidence_matrix, features=None):  # Features can be None
        # Ensure incidence_matrix is a torch tensor
        if isinstance(incidence_matrix, np.ndarray):
            incidence_matrix = torch.tensor(incidence_matrix, dtype=torch.float32)

        D_v_inv_sqrt, D_e_inv = compute_degree_matrices(incidence_matrix)

        # If features are not provided, skip the feature multiplication
        if features is not None:
            if isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32)
            X = torch.matmul(D_v_inv_sqrt, features)
        else:
            # Initialize dummy features based on incidence matrix size
            X = torch.randn(incidence_matrix.shape[0], self.hidden_dim)

        # Embedding generation through the hypergraph propagation
        X = torch.matmul(incidence_matrix.T, X)
        X = torch.matmul(D_e_inv, X)
        X = torch.matmul(incidence_matrix, X)
        X = torch.matmul(D_v_inv_sqrt, X)

        return X

    def generate_soft_labels(self, user_embeddings, temperature=1.0):
        soft_labels = F.softmax(user_embeddings / temperature, dim=1)
        return soft_labels

    def predict(self, incidence_matrix, features=None):
        with torch.no_grad():
            logits = self.forward(incidence_matrix, features)
            _, predicted = torch.max(logits, 1)
        return predicted

# BPR Loss function
def bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings, reg_lambda):
    pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
    neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)
    
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    reg_loss = reg_lambda * (user_embeddings.norm(2).pow(2) + 
                             pos_item_embeddings.norm(2).pow(2) + 
                             neg_item_embeddings.norm(2).pow(2))
    
    return loss + reg_loss

# Negative sampling function
def sample_negative_items(user_item_matrix, num_negatives=1):
    num_users, num_items = user_item_matrix.shape
    negative_samples = []

    for user in range(num_users):
        pos_items = torch.where(user_item_matrix[user] == 1)[0]        
        neg_items = torch.tensor(list(set(range(num_items)) - set(pos_items.tolist())))
        
        if len(neg_items) == 0:
            print(f"No negative items available for user {user}.")
            continue
        
        # Simplified negative sampling logic
        neg_samples = neg_items[torch.randint(0, len(neg_items), (num_negatives,))].tolist()
        
        negative_samples.append(torch.tensor(neg_samples))
    
    if not negative_samples:
        raise ValueError("No negative samples were generated for any users.")
    
    return torch.cat(negative_samples).view(-1)

# Train MLP model with BPR loss and print predictions
def train_mlp_model(mlp_model, hypergraph_model, user_hypergraph_incidence, item_hypergraph_incidence, optimizer, reg_lambda):

    mlp_model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients

    # Get embeddings from the hypergraph model
    user_embeddings = hypergraph_model(user_hypergraph_incidence)  # No features needed
    item_embeddings = hypergraph_model(item_hypergraph_incidence)  # No features needed

    # Ensure user_embeddings and item_embeddings require gradients
    user_embeddings.requires_grad_()
    item_embeddings.requires_grad_()

    assert user_embeddings.shape[1] == item_embeddings.shape[1], "User and item embeddings must have the same feature size."

    # Get positive item indices based on user interactions
    pos_item_indices = user_hypergraph_incidence.nonzero(as_tuple=True)[1]  # Positive items for each user

    # Align user_embeddings with positive item embeddings
    pos_item_embeddings = item_embeddings[pos_item_indices]  # Positive embeddings for all users
    if pos_item_embeddings.shape[0] != user_embeddings.shape[0]:
        pos_item_embeddings = pos_item_embeddings[:user_embeddings.shape[0]]

    # Sample negative items and align with user embeddings
    neg_item_indices = sample_negative_items(user_hypergraph_incidence, num_negatives=1)
    neg_item_embeddings = item_embeddings[neg_item_indices]

    # Calculate BPR loss
    loss = bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings, reg_lambda)

    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the model parameters

    # Predict with the MLP model using user embeddings
    mlp_predictions = mlp_model.predict(user_embeddings)
    # print("MLP Predictions:")
    # print(mlp_predictions)

    return loss.item()

#------------------------------ LightGCN Model---------------------------
# Define LightGCN
class LightGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(LightGCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, adj_matrix):
        x = self.embeddings.weight  # Shape: (num_nodes, embedding_dim)

        # Normalize adjacency matrix
        adj_matrix = adj_matrix + torch.eye(self.num_nodes).to(adj_matrix.device)  # Add self-loops
        rowsum = adj_matrix.sum(1)
        degree_inv_sqrt = rowsum.pow(-0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        norm_adj_matrix = degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt

        # Aggregate embeddings
        x = norm_adj_matrix @ x
        return x

def count_reviews(interactions_df):
    # Aggregate reviews for each user
    bert_user_reviews = interactions_df.groupby('user_id')['review'].apply(list).to_dict()
    
    # Aggregate reviews for each item
    bert_item_reviews = interactions_df.groupby('item_id')['review'].apply(list).to_dict()
    
    return bert_user_reviews, bert_item_reviews

def generate_bert_embeddings(bert_user_reviews, bert_item_reviews, batch_size=8):
    # Load the pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    bert_user_embeddings = {}
    bert_item_embeddings = {}

    # Function to generate embeddings for a batch of reviews
    def get_bert_embedding(bert_reviews_batch):
        # Tokenize the reviews and create tensors
        inputs = tokenizer(bert_reviews_batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the embeddings (mean pooling of last hidden state)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)
        return embeddings.numpy()  # Convert to NumPy array

    # Generate embeddings for users
    for user_id, bert_reviews in bert_user_reviews.items():
        if bert_reviews:  # Check if the user has reviews
            # Process reviews in batches
            bert_user_embeddings[user_id] = []
            for i in range(0, len(bert_reviews), batch_size):
                batch = bert_reviews[i:i + batch_size]
                embedding = get_bert_embedding(batch)
                bert_user_embeddings[user_id].append(embedding)
            # Average the embeddings
            bert_user_embeddings[user_id] = np.mean(np.vstack(bert_user_embeddings[user_id]), axis=0)  # Average embeddings

    # Generate embeddings for items
    for item_id, bert_reviews in bert_item_reviews.items():
        if bert_reviews:  # Check if the item has reviews
            # Process reviews in batches
            bert_item_embeddings[item_id] = []
            for i in range(0, len(bert_reviews), batch_size):
                batch = bert_reviews[i:i + batch_size]
                embedding = get_bert_embedding(batch)
                bert_item_embeddings[item_id].append(embedding)
            # Average the embeddings
            bert_item_embeddings[item_id] = np.mean(np.vstack(bert_item_embeddings[item_id]), axis=0)  # Average embeddings

    return bert_user_embeddings, bert_item_embeddings

# Define the InfoNCE class for contrastive learning
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        # Positive pair logits
        logits = torch.matmul(anchor, positive.T) / self.temperature

        # Negative pair logits
        logits_neg = torch.matmul(anchor, negative.T) / self.temperature

        # Compute loss
        labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)
        positive_loss = F.cross_entropy(logits, labels)
        negative_loss = F.cross_entropy(logits_neg, labels)

        return positive_loss + negative_loss

# Contrastive Learning function with BERT, HypergraphNN, and LightGCN embeddings
def contrastive_learning(bert_user_embeddings, bert_item_embeddings, hgnn_user_embeddings, lightgcn_user_embeddings, 
                         hgnn_item_embeddings, lightgcn_item_embeddings):
    
    user_cl_config = {'temperature': 0.1, 'embedding_dim': 64}
    item_cl_config = {'temperature': 0.2, 'embedding_dim': 64}

    contrastive_loss_user = InfoNCE(temperature=user_cl_config['temperature'])
    contrastive_loss_item = InfoNCE(temperature=item_cl_config['temperature'])

    # Align the number of rows for user embeddings (BERT, HypergraphNN, and LightGCN)
    min_user_size = min(hgnn_user_embeddings.size(0), lightgcn_user_embeddings.size(0), bert_user_embeddings.size(0))
    
    hgnn_user_embeddings = hgnn_user_embeddings[:min_user_size, :]
    lightgcn_user_embeddings = lightgcn_user_embeddings[:min_user_size, :]
    bert_user_embeddings = bert_user_embeddings[:min_user_size, :]

    # Align the number of rows for item embeddings (BERT, HypergraphNN, and LightGCN)
    min_item_size = min(hgnn_item_embeddings.size(0), lightgcn_item_embeddings.size(0), bert_item_embeddings.size(0))
    
    hgnn_item_embeddings = hgnn_item_embeddings[:min_item_size, :]
    lightgcn_item_embeddings = lightgcn_item_embeddings[:min_item_size, :]
    bert_item_embeddings = bert_item_embeddings[:min_item_size, :]

    # Combine BERT, HypergraphNN, and LightGCN embeddings for users
    combined_user_embeddings = torch.cat((hgnn_user_embeddings, lightgcn_user_embeddings, bert_user_embeddings), dim=1)
    
    # Combine BERT, HypergraphNN, and LightGCN embeddings for items
    combined_item_embeddings = torch.cat((hgnn_item_embeddings, lightgcn_item_embeddings, bert_item_embeddings), dim=1)

    # Positive and negative samples for contrastive learning
    positive_user_samples = combined_item_embeddings
    negative_user_samples = torch.cat([combined_item_embeddings[1:], combined_item_embeddings[:1]], dim=0)

    # User contrastive loss
    user_loss = contrastive_loss_user(combined_user_embeddings, positive_user_samples, negative_user_samples)

    positive_item_samples = combined_user_embeddings
    negative_item_samples = torch.cat([combined_user_embeddings[1:], combined_user_embeddings[:1]], dim=0)

    # Item contrastive loss
    item_loss = contrastive_loss_item(combined_item_embeddings, positive_item_samples, negative_item_samples)

    total_loss = user_loss + item_loss

    return combined_user_embeddings, combined_item_embeddings

#--------------------------- Distillation MLP----------------------------

def generate_labels(interactions_df, num_users, num_items):
    # Step 1: Map user_ids and item_ids to unique integers
    user_id_map = {id: idx for idx, id in enumerate(interactions_df['user_id'].unique())}
    item_id_map = {id: idx for idx, id in enumerate(interactions_df['item_id'].unique())}

    # Step 2: Apply the mapping to convert user_id and item_id to integers in the DataFrame
    interactions_df['user_id'] = interactions_df['user_id'].map(user_id_map)
    interactions_df['item_id'] = interactions_df['item_id'].map(item_id_map)

    # Step 3: Initialize the labels matrix (num_users x num_items) with zeros
    labels = np.zeros((num_users, num_items))

    # Step 4: Iterate over the interactions to populate the labels matrix
    for _, row in interactions_df.iterrows():
        user_idx = row['user_id']  # Get the mapped integer index for the user
        item_idx = row['item_id']  # Get the mapped integer index for the item

        # Ensure indices are within the bounds of the matrix
        if user_idx < num_users and item_idx < num_items:
            labels[user_idx, item_idx] = 1  # Set label to 1 where interaction exists

    # Step 5: Convert the labels matrix to a PyTorch tensor
    return torch.tensor(labels, dtype=torch.float32)

class Distillation_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Distillation_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Apply sigmoid to convert logits to probabilities
        return torch.sigmoid(x)

# Distillation Loss: KL Divergence
def distillation_kl_loss(student_logits, teacher_soft_labels, temperature):
    student_soft_labels = F.log_softmax(student_logits / temperature, dim=1)
    kl_loss = F.kl_div(student_soft_labels, teacher_soft_labels, reduction='batchmean') * (temperature ** 2)
    return kl_loss

def bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings, reg_lambda):
    pos_scores = (user_embeddings * pos_item_embeddings).sum(dim=1)
    neg_scores = (user_embeddings * neg_item_embeddings).sum(dim=1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    reg_loss = (user_embeddings.norm(2) ** 2 + pos_item_embeddings.norm(2) ** 2 + neg_item_embeddings.norm(2) ** 2) * reg_lambda
    return loss + reg_loss

def train_mlp_with_distillation(mlp_model, hypergraph_model, user_hypergraph_incidence, 
                                item_hypergraph_incidence, optimizer, reg_lambda, temperature=1.0):
    
    mlp_model.train()  
    optimizer.zero_grad()  

    user_embeddings = hypergraph_model(user_hypergraph_incidence)
    item_embeddings = hypergraph_model(item_hypergraph_incidence)

    # Generate teacher soft labels
    teacher_soft_labels = hypergraph_model.generate_soft_labels(user_embeddings, temperature)

    user_embeddings.requires_grad_()
    item_embeddings.requires_grad_()

    # Positive item sampling
    pos_item_indices = user_hypergraph_incidence.nonzero(as_tuple=True)[1]
    pos_item_embeddings = item_embeddings[pos_item_indices]

    # Negative item sampling
    neg_item_indices = sample_negative_items(user_hypergraph_incidence, num_negatives=1)
    neg_item_embeddings = item_embeddings[neg_item_indices]

    # Ensure the sizes match for BPR loss
    min_size = min(user_embeddings.size(0), pos_item_embeddings.size(0), neg_item_embeddings.size(0))
    
    user_embeddings = user_embeddings[:min_size]
    pos_item_embeddings = pos_item_embeddings[:min_size]
    neg_item_embeddings = neg_item_embeddings[:min_size]

    # Compute BPR loss
    bpr_loss_value = bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings, reg_lambda)

    # Forward pass through MLP
    student_logits = mlp_model(user_embeddings)

    # Compute KL divergence loss for distillation
    kl_loss_value = distillation_kl_loss(student_logits, teacher_soft_labels, temperature)

    # Combine losses
    total_loss = bpr_loss_value + kl_loss_value

    total_loss.backward()
    optimizer.step()

    return total_loss.item()

def create_binary_interaction_matrix(interactions_df):
    users = interactions_df['user_id'].unique()
    items = interactions_df['item_id'].unique()

    user_to_index = {user: idx for idx, user in enumerate(users)}
    item_to_index = {item: idx for idx, item in enumerate(items)}

    interaction_matrix = np.zeros((len(users), len(items)), dtype=int)

    for _, row in interactions_df.iterrows():
        user_idx = user_to_index[row['user_id']]
        item_idx = item_to_index[row['item_id']]
        interaction_matrix[user_idx, item_idx] = 1  # Set to 1 if the user reviewed the item

    return interaction_matrix

def generate_recommendations(mlp_model, hypergraph_model, user_hypergraph_incidence, item_hypergraph_incidence, 
                             interaction_matrix, num_recommendations=5):    
    # Step 1: Obtain user and item embeddings from the hypergraph model
    user_embeddings = hypergraph_model(user_hypergraph_incidence)
    item_embeddings = hypergraph_model(item_hypergraph_incidence)
    
    # Step 2: Generate predictions for all items for each user using the MLP model
    with torch.no_grad():
        user_probabilities = mlp_model(user_embeddings)  # Shape: (num_users, num_items)

    # Step 3: Generate recommendations for each user
    recommendations = {}
    
    for user_idx in range(interaction_matrix.shape[0]):
        user_interactions = interaction_matrix[user_idx]  
        already_interacted = np.where(user_interactions == 1)[0]  
        
        # Get the predicted probabilities for all items for the user
        user_pred_probs = user_probabilities[user_idx].numpy()
        
        # Exclude already interacted items by setting their probabilities to 0
        user_pred_probs[already_interacted] = 0
        
        # Step 5: Recommend top-N items based on predicted probabilities
        top_item_indices = np.argsort(user_pred_probs)[-num_recommendations:][::-1]  # Top-N items
        
        recommendations[user_idx] = top_item_indices
    
    return recommendations

def measure_inference_time(model, input_data):
    """Function to measure inference time."""
    model.eval()  # Set the model to evaluation mode
    start_time = time.time()
    
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_data)

    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return inference_time, output

def calculate_inference_time(model, input_data):
    """Function to calculate inference time for a given model and input data."""
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        predictions = model(input_data)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return predictions, inference_time

# Function to calculate hit rate at k
def hit_rate_at_k(predicted_recommendations, actual_items, k):
    hit_rate_count = 0
    total_users = len(predicted_recommendations)

    # Loop through each user and their recommendations
    for user_idx in range(total_users):
        top_recommended_items = set(predicted_recommendations[user_idx][:k])
        
        # Check if there is any intersection between actual items and recommended items
        if actual_items[user_idx].intersection(top_recommended_items):
            hit_rate_count += 1

    return hit_rate_count / total_users if total_users > 0 else 0

def evaluate_recommendations(predicted_recommendations, actual_items, top_n):
    precision_list = []
    recall_list = []
    f1_list = []

    for user_idx in range(predicted_recommendations.shape[0]):
        predicted = predicted_recommendations[user_idx]  # Get recommendations for this user
        actual = actual_items[user_idx]  # Get actual items for this user (make sure actual_items is indexed properly)

        # Get the top_n predicted recommendations
        predicted_items = set(predicted[:top_n])  # Adjust this if predicted is not sorted

        # Calculate precision, recall, and F1
        true_positives = len(predicted_items & actual)
        precision = true_positives / len(predicted_items) if predicted_items else 0
        recall = true_positives / len(actual) if actual else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

def evaluate_recommendations_multiple_runs(predicted_recommendations, actual_items, top_n, num_runs=10):
    precisions = []
    recalls = []
    f1_scores = []
    hit_rates = []  

    for _ in range(num_runs):
        precision, recall, f1 = evaluate_recommendations(predicted_recommendations, actual_items, top_n)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Calculate hit rate at k=10
        hit_rate = hit_rate_at_k(predicted_recommendations, actual_items, k=10)
        hit_rates.append(hit_rate)

    # Calculate mean and standard deviation for each metric
    precision_mean = np.mean(precisions)
    precision_std = np.std(precisions)
    recall_mean = np.mean(recalls)
    recall_std = np.std(recalls)
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    hit_rate_mean = np.mean(hit_rates)  # Mean hit rate
    hit_rate_std = np.std(hit_rates)     # Std hit rate

    return precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, hit_rate_mean, hit_rate_std

def main():
    # Load interactions data
    interactions_path = 'Dataset Address'
    interactions_df = pd.read_csv(interactions_path)

    # Split the data into training (70%), validation (20%), and test sets (10%)
    train_data, temp_data = train_test_split(interactions_df, test_size=0.3, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.333, random_state=42)  

    # Create hypergraphs and incidence matrices based on training data
    user_hypergraph_incidence, item_hypergraph_incidence, _, _ = create_hypergraphs_and_incidence_matrices(train_data)

    # Create adjacency matrix for LightGCN based on training data
    adj_matrix, node_to_index = create_homogeneous_adjacency_matrix(train_data)

    # Define number of nodes from the adjacency matrix
    num_nodes = adj_matrix.shape[0]

    # Initialize the models
    lightgcn_model = LightGCN(num_nodes=num_nodes, embedding_dim=64)
    hypergraph_model = HypergraphNN(input_dim=user_hypergraph_incidence.shape[1], hidden_dim=64, output_dim=64)
    mlp_model = MLP(input_dim=64, hidden_dim=64, output_dim=64)  # Match output_dim with HypergraphNN output

    # Optimizer
    optimizer = torch.optim.Adam(list(hypergraph_model.parameters()) + list(mlp_model.parameters()), lr=0.001)

    # Training loop for MLP with distillation
    for epoch in range(200):
        loss = train_mlp_with_distillation(mlp_model, hypergraph_model, user_hypergraph_incidence, item_hypergraph_incidence, optimizer, reg_lambda=0.001)

    # Generate HypergraphNN embeddings
    hgnn_user_embeddings = hypergraph_model(user_hypergraph_incidence)  # User embeddings
    hgnn_item_embeddings = hypergraph_model(item_hypergraph_incidence)  # Item embeddings

    # Generate predictions from MLP using the embeddings from HypergraphNN
    mlp_predictions = mlp_model(hgnn_user_embeddings)

    # Generate LightGCN embeddings
    lightgcn_user_embeddings = lightgcn_model(adj_matrix)  # LightGCN user embeddings
    lightgcn_item_embeddings = lightgcn_user_embeddings  # Assuming the same for items

    # Load or simulate BERT embeddings
    bert_user_embeddings = torch.tensor(np.random.rand(30, 768), dtype=torch.float32)  
    bert_item_embeddings = torch.tensor(np.random.rand(30, 768), dtype=torch.float32)  

    # Call contrastive learning with generated embeddings
    combined_user_embeddings, combined_item_embeddings = contrastive_learning(
        bert_user_embeddings=bert_user_embeddings,
        bert_item_embeddings=bert_item_embeddings,
        hgnn_user_embeddings=hgnn_user_embeddings,
        lightgcn_user_embeddings=lightgcn_user_embeddings,
        hgnn_item_embeddings=hgnn_item_embeddings,
        lightgcn_item_embeddings=lightgcn_item_embeddings
    )

    # Define the Distillation_MLP model
    embedding_dim = combined_user_embeddings.size(1) + combined_item_embeddings.size(1)
    hidden_dim = 128  # Adjustable hidden dimension
    output_dim = 64   # Ensure this matches the teacher soft labels

    mlp_Distill = Distillation_MLP(input_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Prepare input for Distillation_MLP
    user_item_pairs = []
    for user_embedding, item_embedding in zip(combined_user_embeddings, combined_item_embeddings):
        combined_input = torch.cat((user_embedding, item_embedding), dim=0)
        user_item_pairs.append(combined_input)

    # Convert list to tensor
    user_item_pairs = torch.stack(user_item_pairs).float()

    # Measure inference time for Distillation_MLP predictions
    predictions_distill, distill_inference_time = calculate_inference_time(mlp_Distill, user_item_pairs)
    print(f"Distillation MLP Inference Time: {distill_inference_time:.2f} ms")

    # Get predicted interaction scores
    predicted_interaction_scores = mlp_Distill(user_item_pairs).detach().numpy()

    # Reshape predicted interaction scores into (num_users, num_items)
    num_users = combined_user_embeddings.size(0)
    num_items = predicted_interaction_scores.size // num_users  # Ensure it divides correctly
    predicted_interaction_matrix = predicted_interaction_scores.reshape(num_users, num_items)

    # Generate recommendations based on the training data
    top_n = 5  # Number of top recommendations per user
    recommendations = np.argsort(-predicted_interaction_matrix, axis=1)[:, :top_n]

    # Print the recommendations for each user
    for user_idx, recommended_items in enumerate(recommendations):
        print(f"User {user_idx}: Recommended items: {recommended_items}")

    # Evaluate recommendations on validation set
    validation_actual_items = validation_data.groupby('user_id')['item_id'].apply(set).tolist()
    precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, hit_rate_mean, hit_rate_std = evaluate_recommendations_multiple_runs(recommendations, validation_actual_items, top_n)
    
    # Evaluate recommendations on test set
    test_actual_items = test_data.groupby('user_id')['item_id'].apply(set).tolist()
    precision_mean_test, precision_std_test, recall_mean_test, recall_std_test, f1_mean_test, f1_std_test, hit_rate_mean_test, hit_rate_std_test = evaluate_recommendations_multiple_runs(recommendations, test_actual_items, top_n)


if __name__ == "__main__":
    main()
