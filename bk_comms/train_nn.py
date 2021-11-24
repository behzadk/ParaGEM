import torch
import torch.nn.functional as F
import cobra
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler


class NeuralNet(torch.nn.Module):
    def __init__(self, n_features, n_outputs, hidden_width=128, n_hidden_layers=1):
        super(NeuralNet, self).__init__()

        modules = nn.ModuleList()

        blocks = []

        # Input layer
        input_block = nn.Sequential(
            nn.Linear(n_features, hidden_width),
            nn.ReLU(),
            # nn.Dropout(p=0.2)
        )

        blocks.append(input_block)

        for i in range(n_hidden_layers - 1):
            new_block = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            # nn.Dropout(p=0.2)
            )

            blocks.append(new_block)


        output_block = nn.Sequential(nn.Linear(hidden_width, n_outputs))

        blocks.append(output_block)

        self.layers = nn.Sequential(*blocks)        
        
        # for idx, l in enumerate(self.layers):
        #     print(self.layers[idx])
        #     if isinstance(self.layers[idx], nn.Linear):
        #         print("weight")
        #         self.layers[idx].weight.data.normal_(0.0, 1e-5)


    def forward(self, x):
        return self.layers(x)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_data, label_data):
        'Initialization'
        self.inputs = input_data
        self.labels = label_data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.inputs[idx]
        y = self.labels[idx]

        return {'X': X, 'y': y}



class FluxBalanceNN:
    def __init__(self, population):
        self.population = population
        self.dynamic_compounds = population.dynamic_compounds

        self.n_features = self.get_n_features()
        self.n_outputs = self.get_n_outputs()

        self.loss_func = torch.nn.MSELoss()
    
    def initialize_writer(self, comment=None):
        self.writer = SummaryWriter(comment=comment)


    def initialize_model(self):
        self.nn_model = NeuralNet(n_features=self.n_features, n_outputs=self.n_outputs)

    def initialize_optimizer(self, lr):
        self.optimizer = torch.optim.AdamW(self.nn_model.parameters(), lr=lr)


    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return epoch

    def save_checkpoint(self, path, epoch, avg_test_loss, avg_test_error):
        torch.save({
        'epoch': epoch,
        'model_state_dict': self.nn_model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'avg_test_loss': avg_test_loss,
        'avg_test_error': avg_test_error,
        }, path)

    def get_n_features(self):
        return len(self.dynamic_compounds)

    def get_n_outputs(self):
        # One extra for the biomass flux
        return len(self.dynamic_compounds) + 1

    def generate_constraints_dataset(self, n_batches=10000, batch_size=5, samples_scale='log_uniform', null_mask_probability=0.0, min_uptake=1e-40, max_uptake=10):
        # Generate constraint inputs
        # Calculate FBA output
        if samples_scale == 'log_uniform':
            X = np.exp(np.random.uniform(np.log(min_uptake), np.log(max_uptake), 
            size=[n_batches, batch_size, self.n_features]))
            X = X * -1
        
        elif samples_scale == 'uniform':
            X = (np.random.uniform(min_uptake, max_uptake, 
            size=[n_batches, batch_size, self.n_features]))
            X = X * -1


        y = np.zeros(shape=[n_batches, batch_size, self.n_outputs])

        for batch_idx in range(n_batches):
            for vector_idx in range(batch_size):
                # print(batch_idx, vector_idx)
                x = X[batch_idx][vector_idx]
                
                proportion_null = np.random.uniform(0, null_mask_probability)
                mask = np.random.choice([True, False], size=(len(x)), p=[1.0 - proportion_null, proportion_null])

                x = mask * x
                X[batch_idx][vector_idx] = x

                self.population.update_reaction_constraints(x)
                self.population.optimize()
                
                # Compound fluxes have some zero values
                compound_fluxes = self.population.get_dynamic_compound_fluxes()
                biomass_flux = self.population.get_growth_rate()

                y[batch_idx][vector_idx][0] = biomass_flux
                y[batch_idx][vector_idx][1:] = compound_fluxes

                if np.isnan(y).any():
                    print(batch_idx, vector_idx, "y has nan")
                    print(x)
                    exit()

                elif np.isnan(x).any():
                    print(batch_idx, vector_idx, "x has nan")
                    exit()

        return X, y

    def train(self, epochs, batch_size):
        max_epochs = epochs
        train_ldr = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size, 
            shuffle=True)

        test_ldr = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size, 
            shuffle=False)

        for epoch in range(max_epochs):
            total_train_loss = 0

            # Train loop
            for (batch_idx, batch) in enumerate(train_ldr):
                X = batch['X']
                y = batch['y']

                train_loss = self.train_loop(epoch, X, y)
                total_train_loss += train_loss
            
            avg_train_loss = total_train_loss / len(train_ldr)
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)

            if epoch % 10 == 0:
                total_val_loss = 0
                total_val_error = 0
                # Validation against all test set
                for (batch_idx, batch) in enumerate(test_ldr):
                    X = batch['X']
                    y = batch['y']

                    val_loss, val_error = self.val_loop(epoch, X, y, inv_transform_prediction=False)
                    total_val_loss += val_loss
                    total_val_error += val_error
                    output_path = './test_nn_checkpoints/model.pt'

                avg_val_error = total_val_error / len(test_ldr)
                self.writer.add_scalar("avg_test_error", avg_val_error, epoch)

                total_recall_loss = 0
                total_recall_error = 0
                # Recall against all test set
                for (batch_idx, batch) in enumerate(train_ldr):
                    X = batch['X']
                    y = batch['y']
                    val_loss, val_error = self.val_loop(epoch, X, y, inv_transform_prediction=False)
                    total_recall_loss += val_loss
                    total_recall_error += val_error
                    output_path = './test_nn_checkpoints/model.pt'

                avg_recall_error = total_recall_error / len(test_ldr)
                self.writer.add_scalar("avg_recall_error", avg_recall_error, epoch)

                self.save_checkpoint(output_path, epoch, avg_train_loss, avg_val_error)

    def fit_labels_scaler(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_dataset.dataset[:]['y'])

    def transform_labels(self, labels):
        return self.scaler.transform(labels)

    def transform_train_dataset_labels(self):
        self.train_dataset.dataset.labels = self.transform_labels(self.train_dataset.dataset.labels)

    def inverse_transform_scaler(self, x):
        return self.scaler.inverse_transform(x)

    def load_dataset(self, data_dir, data_suffixes, head=None, single_column=None):
        X_arr = []
        y_arr = []
        for suffix in data_suffixes:
            input_data = f'{data_dir}/train_data_{suffix}.csv'
            label_data = f'{data_dir}/train_labels_{suffix}.csv'
            
            X = np.loadtxt(input_data, delimiter=',')
            y = np.loadtxt(label_data, delimiter=',')
            
            X_arr.append(X.astype(np.float32))

            if single_column is not None:
                y = y[:, single_column].reshape(-1, 1)
                y_arr.append(y.astype(np.float32))
            
            else:
                y_arr.append(y.astype(np.float32))

        if head:
            self.dataset = Dataset(np.concatenate(X_arr)[:head], np.concatenate(y_arr)[:head])

        else:
            self.dataset = Dataset(np.concatenate(X_arr), np.concatenate(y_arr))
    
    def generate_train_val_split(self, test_prop):
        data_len = len(self.dataset)

        n_val = int(data_len * test_prop)
        n_train = data_len - n_val
        print("train: ", n_train, "validation: ", n_val)

        train_set, val_set = torch.utils.data.random_split(self.dataset, lengths=[n_train, n_val])

        return train_set, val_set

    def set_train_dataset(self, train_set):
        self.train_dataset = train_set
    
    def set_validation_dataset(self, val_set):
        self.val_dataset = val_set

    def train_loop(self, epoch, X, y):
        self.optimizer.zero_grad()

        # Compute prediction and loss
        pred = self.nn_model(X)
        loss = self.loss_func(pred, y)
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()

        return loss

    def val_loop(self, epoch, X, y, inv_transform_prediction=False):
        test_loss, test_abs_error = 0, 0
        num_batches = len(self.val_dataset)
        size = len(y)

        with torch.no_grad():
            pred = self.nn_model(X)
            val_loss = self.loss_func(pred, y)

            if inv_transform_prediction:
                pred = self.inverse_transform_scaler(pred)

            val_error = torch.sum(torch.abs(y - pred))

        return val_loss, val_error


        

