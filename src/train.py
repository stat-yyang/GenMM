# train.py

import torch
import torch.nn as nn
import torch.optim as optim

from utils import create_graph_data

def train_model(encoder, generators, gnn_model, dataloader, device, num_epochs=10, lambda_rec=1.0, learning_rate=0.001):
    encoder.to(device)
    for gen in generators:
        gen.to(device)
    gnn_model.to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) +
        [param for gen in generators for param in gen.parameters()] +
        list(gnn_model.parameters()),
        lr=learning_rate
    )
    criterion_rec = nn.MSELoss()
    criterion_pred = nn.MSELoss()  # Assuming regression task

    for epoch in range(num_epochs):
        total_loss = 0.0
        encoder.train()
        gnn_model.train()
        for batch in dataloader:
            X_batch = batch['X'].to(device)
            N_list_batch = [N.to(device) for N in batch['N_list']]
            M_list_batch = [M.to(device) for M in batch['M_list']]
            Y_batch = batch['Y'].to(device)

            # Encode X to Z
            Z = encoder(X_batch)

            # Generate Modalities
            hat_N_list = [gen(Z) for gen in generators]

            # Combine Real and Generated Modalities
            tilde_N_list = []
            rec_loss = 0.0
            for N_real, N_gen, M in zip(N_list_batch, hat_N_list, M_list_batch):
                M = M.float()
                tilde_N = M * N_real + (1 - M) * N_gen
                tilde_N_list.append(tilde_N)
                rec_loss += criterion_rec(M * N_gen, M * N_real)

            # Prepare data for GNN
            modalities_combined = torch.cat(tilde_N_list, dim=1)
            graph_data = create_graph_data(modalities_combined)
            graph_data = graph_data.to(device)

            # Predict Y with GNN
            Y_pred = gnn_model(graph_data.x, graph_data.edge_index)

            # Compute Prediction Loss
            pred_loss = criterion_pred(Y_pred, Y_batch)

            # Total Loss
            loss = lambda_rec * rec_loss + pred_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
