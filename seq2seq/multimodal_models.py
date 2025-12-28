import torch
import torch.nn as nn
import torch.utils.data as tud
import torch.nn.functional as F
from timeit import default_timer as timer
from tqdm.auto import tqdm
import pandas as pd
import warnings
import copy
from typing import Any, Callable, Optional, Union

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.best_epoch = None

    def __call__(self, val_loss, model, current_epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_epoch = current_epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_epoch = current_epoch
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

class MultimodalSeq2Seq(nn.Module): 
    """
    A generic sequence-to-sequence model. All other sequence-to-sequence models should extend this class 
    with a __init__ and forward methods, in the same way as in normal PyTorch.
    """
    def print_architecture(self):
        """
        Displays the information about the model in standard output. 
        """
        for k in self.architecture.keys():
            print(f"{k.replace('_', ' ').capitalize()}: {self.architecture[k]}")
        print(f"Trainable parameters: {sum([p.numel() for p in self.parameters()]):,}")
        print()
        
    def fit(self, 
            X_train_smiles, 
            X_train_descriptors,
            Y_train, 
            X_dev_smiles = None, 
            X_dev_descriptors = None, 
            Y_dev = None, 
            batch_size = 100, 
            epochs = 5, 
            learning_rate = 10**-4, 
            weight_decay = 0, 
            progress_bar = 0, 
            save_path = None):
        """
        A generic training method with Adam and Cross Entropy.

        Parameters
        ----------    
        X_train_smiles: LongTensor of shape (train_examples, train_input_length)
            The input sequences of the training set.
            
        X_train_descriptors: LongTensor of shape (train_examples, train_descriptors_length)
            The molecular descriptors of the input sequences of the training set.
            
        Y_train: LongTensor of shape (train_examples, train_output_length)
            The output sequences of the training set.
            
        X_dev_smiles: LongTensor of shape (dev_examples, dev_input_length), optional
            The input sequences for the development set.
            
        X_dev_descriptors: LongTensor of shape (dev_examples, dev_descriptors_length), optional
            The molecular descriptors of the input sequences for the development set.
            
        Y_dev: LongTensor of shape (dev_examples, dev_output_length), optional
            The output sequences for the development set.
            
        batch_size: int
            The number of examples to process in each batch.

        epochs: int
            The number of epochs of the training process.
            
        learning_rate: float
            The learning rate to use with Adam in the training process. 
            
        weight_decay: float
            The weight_decay parameter of Adam (L2 penalty), useful for regularizing models. For a deeper 
            documentation, go to https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam            

        progress_bar: int
            Shows a tqdm progress bar, useful for tracking progress with large tensors.
            If equal to 0, no progress bar is shown. 
            If equal to 1, shows a bar with one step for every epoch.
            If equal to 2, shows the bar when equal to 1 and also shows a bar with one step per batch for every epoch.
            If equal to 3, shows the bars when equal to 2 and also shows a bar to track the progress of the evaluation
            in the development set.
            
        save_path: string, optional
            Path to save the .pt file containing the model parameters when the training ends.

        Returns
        -------
        performance: Pandas DataFrame
            DataFrame with the following columns: epoch, train_loss, train_error_rate, (optionally dev_loss and 
            dev_error_rate), minutes, learning_rate, weight_decay, model, encoder_embedding_dimension, 
            decoder_embedding_dimension, encoder_hidden_units, encoder_layers, decoder_hidden_units, decoder_layers, 
            dropout, parameters and one row for each of the epochs, containing information about the training process.
        """
        assert X_train_smiles.shape[0] == X_train_descriptors.shape[0] == Y_train.shape[0]
        
        if (X_dev_smiles is not None and X_dev_descriptors is not None and Y_dev is not None):
            assert X_dev_smiles.shape[0] == X_dev_descriptors.shape[0] == Y_dev.shape[0]
            dev = True
        else:
            dev = False
        train_dataset = tud.TensorDataset(X_train_smiles, X_train_descriptors, Y_train)
        train_loader = tud.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        criterion = nn.CrossEntropyLoss(ignore_index = 0)
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate, weight_decay = weight_decay)
        self.early_stopping = EarlyStopping(patience=5, delta=0.0)
        performance = []
        start = timer()
        epochs_iterator = range(1, epochs + 1)
        if progress_bar > 0:
            epochs_iterator = tqdm(epochs_iterator)
        print("Training started")
        print("X_train.shape:", X_train_smiles.shape)
        print("Y_train.shape:", Y_train.shape)
        if dev:
            print("X_dev.shape:", X_dev_smiles.shape)
            print("Y_dev.shape:", Y_dev.shape)
        print(f"Epochs: {epochs:,}\nLearning rate: {learning_rate}\nWeight decay: {weight_decay}")
        header_1 = "Epoch | Train                "
        header_2 = "      | Loss     | Error Rate"
        rule = "-" * 29
        if dev:
            header_1 += " | Development          "
            header_2 += " | Loss     | Error Rate"
            rule += "-" * 24
        header_1 += " | Minutes"
        header_2 += " |"
        rule += "-" * 10
        print(header_1, header_2, rule, sep = "\n")
        for e in epochs_iterator:
            self.train()
            losses = []
            errors = []
            sizes = []
            train_iterator = train_loader
            if progress_bar > 1:
                train_iterator = tqdm(train_iterator)
            for x_smiles, x_descriptors, y in train_iterator:
                # compute loss and backpropagate
                probabilities = self.forward(x_smiles, x_descriptors, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]
                loss = criterion(probabilities, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # compute accuracy
                predictions = probabilities.argmax(1)
                batch_errors = (predictions != y)
                # append the results
                losses.append(loss.item())
                errors.append(batch_errors.sum().item())
                sizes.append(batch_errors.numel())
            train_loss = sum(losses) / len(losses)
            train_error_rate = 100 * sum(errors) / sum(sizes)
            t = (timer() - start) / 60
            status_string = f"{e:>5} | {train_loss:>8.4f} | {train_error_rate:>10.3f}"
            status = {"epoch":e,
                      "train_loss": train_loss,
                      "train_error_rate": train_error_rate}
            if dev:
                dev_loss, dev_error_rate = self.evaluate(X_dev_smiles,
                                                         X_dev_descriptors,
                                                         Y_dev, 
                                                         batch_size = batch_size, 
                                                         progress_bar = progress_bar > 2, 
                                                         criterion = criterion)
                status_string += f" | {dev_loss:>8.4f} | {dev_error_rate:>10.3f}"
                status.update({"dev_loss": dev_loss, "dev_error_rate": dev_error_rate})
                self.early_stopping(dev_loss, self, current_epoch = e)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

                self.early_stopping.load_best_model(self)
                torch.save(self.early_stopping.best_model_state, "best_multimodalmodel.pth")
            status.update({"training_minutes": t,
                           "learning_rate": learning_rate,
                           "weight_decay": weight_decay})
            performance.append(status)
            if save_path is not None:  
                if (not dev) or (e < 2) or (dev_loss < min([p["dev_loss"] for p in performance[:-1]])):
                    torch.save(self.state_dict(), save_path)
            status_string += f" | {t:>7.1f}"
            print(status_string)
        print()
        return pd.concat((pd.DataFrame(performance), 
                          pd.DataFrame([self.architecture for i in performance])), axis = 1)\
               .drop(columns = ["source_index", "target_index"])
    
            
    def evaluate(self, 
                 X_smiles, 
                 X_descriptors,
                 Y, 
                 criterion = nn.CrossEntropyLoss(), 
                 batch_size = 128, 
                 progress_bar = False):
        """
        Evaluates the model on a dataset.
        
        Parameters
        ----------
        X_smiles: LongTensor of shape (examples, input_length)
            The input sequences of the dataset.

        X_descriptors: LongTensor of shape (examples, descriptors_dimension)
            The molecular descriptor of the input sequences of the dataset.
            
        Y: LongTensor of shape (examples, output_length)
            The output sequences of the dataset.
            
        criterion: PyTorch module
            The loss function to evalue the model on the dataset, has to be able to compare self.forward(X, Y) and Y
            to produce a real number.
            
        batch_size: int
            The batch size of the evaluation loop.
            
        progress_bar: bool
            Shows a tqdm progress bar, useful for tracking progress with large tensors.
            
        Returns
        -------
        loss: float
            The average of criterion across the whole dataset.
            
        error_rate: float
            The step-by-step accuracy of the model across the whole dataset. Useful as a sanity check, as it should
            go to zero as the loss goes to zero.
            
        """
        dataset = tud.TensorDataset(X_smiles, X_descriptors, Y)
        loader = tud.DataLoader(dataset, batch_size = batch_size)
        self.eval()
        losses = []
        errors = []
        sizes = []
        with torch.no_grad():
            iterator = iter(loader)
            if progress_bar:
                iterator = tqdm(iterator)
            for batch in iterator:
                x_smiles, x_descriptors, y = batch
                # compute loss
                probabilities = self.forward(x_smiles, x_descriptors, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]
                loss = criterion(probabilities, y)
                # compute accuracy
                predictions = probabilities.argmax(1)
                batch_errors = (predictions != y)
                # append the results
                losses.append(loss.item())
                errors.append(batch_errors.sum().item())
                sizes.append(batch_errors.numel())
            loss = sum(losses) / len(losses)
            error_rate = 100 * sum(errors) / sum(sizes)
        return loss, error_rate 

class MultimodalTransformer(MultimodalSeq2Seq):
    def __init__(self, 
                 source_index, 
                 target_index,
                 max_sequence_length = 32,
                 embedding_dimension = 32,
                 descriptors_dimension = 907,
                 feedforward_dimension = 128,
                 encoder_layers = 2,
                 decoder_layers = 2,
                 attention_heads = 2,
                 activation = "relu",
                 dropout = 0.0):
        """
        A PyTorch implementation of a Transformer model that learns from sequences and molecular descriptors.
        
        Parameters
        ----------
        in_vocabulary: dictionary
            Vocabulary with the index:token pairs for the inputs of the model.
            
        out_vocabulary: dictionary
            Vocabulary with the token:index pairs for the outputs of the model.
            
        max_sequence_length: int
            Maximum sequence length accepted by the model, both for the encoder and the decoder.
            
        embedding_dimension: int
            Dimension of the embeddings of the model.
            
        descriptors_dimension: int
            Dimension of the molecular descriptors per compound.
            
        feedforward_dimension: int
            Dimension of the feedforward network inside the self-attention layers of the model.
            
        encoder_layers: int
            Hidden layers of the encoder.
            
        decoder_layers: int
            Hidden layers of the decoder.
            
        attention_heads: int
            Attention heads inside every self-attention layer of the model.
            
        activation: string
            Activation function of the feedforward network inside the self-attention layers of the model. Can
            be either 'relu' or 'gelu'.
            
        dropout: float between 0.0 and 1.0
            Dropout rate to apply to whole model.
        """
        super().__init__()
        self.source_index = source_index
        self.target_index = target_index
        self.source_embeddings = nn.Embedding(len(source_index), embedding_dimension)
        self.target_embeddings = nn.Embedding(len(target_index), embedding_dimension)
        self.positional_embeddings = nn.Embedding(max_sequence_length, embedding_dimension)
        self.fusion_layer = nn.Linear(descriptors_dimension + embedding_dimension, embedding_dimension)
        self.transformer = nn.Transformer(d_model = embedding_dimension, 
                                          dim_feedforward = feedforward_dimension,
                                          nhead = attention_heads, 
                                          num_encoder_layers = encoder_layers, 
                                          num_decoder_layers = decoder_layers,
                                          activation = activation,
                                          dropout = dropout)
        self.output_layer = nn.Linear(embedding_dimension, len(target_index))
        self.architecture = dict(model = "Seq2Seq Multimodal Transformer",
                                 source_index = source_index,
                                 target_index = target_index,
                                 max_sequence_length = max_sequence_length,
                                 embedding_dimension = embedding_dimension,
                                 descriptors_dimension = descriptors_dimension,
                                 feedforward_dimension = feedforward_dimension,
                                 encoder_layers = encoder_layers,
                                 decoder_layers = decoder_layers,
                                 attention_heads = attention_heads,
                                 activation = activation,
                                 dropout = dropout)
        self.print_architecture()

    def forward(self, X_smiles, X_descriptors, Y):
        """
        Forward method of the model.
        
        Parameters
        ----------
        X_smiles: LongTensor of shape (batch_size, input_length)
            Tensor of integers containing the inputs for the model.

        X_descriptors: LongTensor of shape (batch_size, descriptors_dimension)
            Tensor of integers containing the descriptors for the model.
            
        Y: LongTensor of shape (batch_size, output_length)
            Tensor of integers containing the output produced so far.
            
        Returns
        -------
        output: FloatTensor of shape (batch_size, output_length, len(out_vocabulary))
            Tensor of floats containing the inputs for the final Softmax layer (usually integrated in the loss function).
        """
        assert X_smiles.shape[1] <= self.architecture["max_sequence_length"]
        assert Y.shape[1] <= self.architecture["max_sequence_length"]
        X_smiles = self.source_embeddings(X_smiles)
        
        X_smiles_positional = torch.arange(X_smiles.shape[1], device = X_smiles.device).repeat((X_smiles.shape[0], 1))
        X_smiles_positional = self.positional_embeddings(X_smiles_positional)
        X_smiles = (X_smiles + X_smiles_positional).transpose(0, 1)

        X_descriptors = X_descriptors.unsqueeze(0)
        X_descriptors = X_descriptors.expand([X_smiles.shape[0], -1, -1])
        X = torch.cat([X_smiles, X_descriptors], dim=-1)

        X = self.fusion_layer(X)
        
        Y = self.target_embeddings(Y)
        Y_positional = torch.arange(Y.shape[1], device = Y.device).repeat((Y.shape[0], 1))
        Y_positional = self.positional_embeddings(Y_positional)
        Y = (Y + Y_positional).transpose(0, 1)
        mask = self.transformer.generate_square_subsequent_mask(Y.shape[0]).to(Y.device)
        transformer_output = self.transformer.forward(src = X,
                                                      tgt = Y, 
                                                      tgt_mask = mask)
        transformer_output = transformer_output.transpose(0, 1)
        return self.output_layer(transformer_output)

class MultimodalBiLSTM(MultimodalSeq2Seq):
    def __init__(self, 
                 source_index, 
                 target_index, 
                 encoder_embedding_dimension,
                 descriptors_dimension,
                 decoder_embedding_dimension,
                 encoder_hidden_units, 
                 encoder_layers,
                 decoder_hidden_units,
                 decoder_layers,
                 dropout):
        """
        A Seq2Seq Bidirectional LSTM model that learns from sequences and molecular descriptors.
        
        Parameters
        ----------
        in_vocabulary: dictionary
            Vocabulary with the index:token pairs for the inputs of the model.
            
        out_vocabulary: dictionary
            Vocabulary with the token:index pairs for the outputs of the model.
            
        encoder_embedding_dimension: int
            Dimension of the embeddings to feed into the encoder.

        descriptors_dimension: int
            Dimension of the molecular descriptors per compound.
            
        decoder_embedding_dimension: int
            Dimension of the embeddings to feed into the decoder.
            
        encoder_hidden_units: int
            Hidden size of the encoder.
            
        encoder_layers: int
            Hidden layers of the encoder.
            
        decoder_hidden_units: int
            Hidden units of the decoder.
            
        decoder_layers: int
            Hidden layers of the decoder.
            
        dropout: float between 0.0 and 1.0
            Dropout rate to apply to whole model.
        """
        self.source_index = source_index
        self.target_index = target_index
        self.descriptors_dimension = descriptors_dimension
        super().__init__()
        self.source_embeddings = nn.Embedding(len(source_index), encoder_embedding_dimension)
        self.target_embeddings = nn.Embedding(len(target_index), decoder_embedding_dimension)
        self.encoder_rnn = nn.LSTM(input_size = encoder_embedding_dimension, 
                                   hidden_size = encoder_hidden_units, 
                                   num_layers = encoder_layers,
                                   dropout = dropout,
                                   bidirectional = True)
        self.fusion_layer = nn.Linear(encoder_layers * encoder_hidden_units + descriptors_dimension, encoder_layers * encoder_hidden_units)
        self.decoder_rnn = nn.LSTM(input_size = encoder_layers * encoder_hidden_units + decoder_embedding_dimension, 
                                   hidden_size = decoder_hidden_units, 
                                   num_layers = decoder_layers,
                                   dropout = dropout)
        self.output_layer = nn.Linear(decoder_hidden_units, len(target_index))
        self.architecture = dict(model = "Seq2Seq Multimodal Bi-LSTM",
                                 source_index = source_index, 
                                 target_index = target_index, 
                                 encoder_embedding_dimension = encoder_embedding_dimension,
                                 descriptors_dimension = descriptors_dimension,
                                 decoder_embedding_dimension = decoder_embedding_dimension,
                                 encoder_hidden_units = encoder_hidden_units, 
                                 encoder_layers = encoder_layers,
                                 decoder_hidden_units = decoder_hidden_units,
                                 decoder_layers = decoder_layers,
                                 dropout = dropout)
        self.print_architecture()
        
    def forward(self, X_smiles, X_descriptors, Y):
        """
        Forward method of the model.
        
        Parameters
        ----------
        X_smiles: LongTensor of shape (batch_size, input_length)
            Tensor of integers containing the inputs for the model.

        X_descriptors: LongTensor of shape (batch_size, descriptors_dimension)
            Tensor of integers containing the descriptors for the model.
            
        Y: LongTensor of shape (batch_size, output_length)
            Tensor of integers containing the output produced so far.
            
        Returns
        -------
        output: FloatTensor of shape (batch_size, output_length, len(out_vocabulary))
            Tensor of floats containing the inputs for the final Softmax layer (usually integrated into the loss function).
        """
        X = self.source_embeddings(X_smiles.T)
        #### When bidirectional=True, output (encoder) will contain a concatenation of the forward and reverse hidden states at each time step in the sequence.
        encoder, (encoder_last_hidden, encoder_last_memory) = self.encoder_rnn(X)
        #### For bidirectional LSTMs, h_n is not equivalent to the last element of output; 
        #### the former contains the final forward and reverse hidden states, 
        #### while the latter contains the final forward hidden state and the initial reverse hidden state.
        
        # For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively. 
        # Process bidirectional hidden states
        num_layers, batch_size, hidden_size = encoder_last_hidden.shape
        # Merge bidirectional outputs
        encoder_last_hidden = encoder_last_hidden.view(2, num_layers // 2, batch_size, hidden_size)
        # Sum forward and backward
        encoder_last_hidden = encoder_last_hidden.sum(dim=0)  
        encoder_last_hidden = encoder_last_hidden.transpose(0, 1).flatten(start_dim = 1)
       
        fused_representation = torch.cat([encoder_last_hidden, X_descriptors], dim = -1)
        fused_representation = self.fusion_layer(fused_representation)
        fused_representation = fused_representation.repeat(Y.shape[1], 1, 1)

        Y = self.target_embeddings(Y.T)
        Y = torch.cat((Y, fused_representation), axis=-1) 

        decoder_outputs, (decoder_last_hidden, decoder_last_memory) = self.decoder_rnn(Y)
        output = self.output_layer(decoder_outputs.transpose(0, 1))  

        return output   