import time
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def freeze_noMWE(model):
    for module in model.modules():
        if not "MaxWEnt" in module.__class__.__name__:
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)
            module.eval()
    return model


class MaxWEnt:
    
    def __init__(self, network, lambda_=1., **kwargs):
        self.network = network
        self.lambda_ = lambda_


    def compile(self, optimizer, loss, metrics, clip_grad_norm=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.clip_grad_norm = clip_grad_norm


    def __call__(self, inputs, clip=None, seed=None):
        device = next(self.network.parameters()).device
        inputs = inputs.to(device)
        if clip is not None:
            self._update_clip_in_layers(clip)
        if seed is not None:
            self._update_seed_in_layers(seed)
        self.network.eval()
        out = self.network(inputs)
        if clip is not None:
            self._update_clip_in_layers(None)
        if seed is not None:
            self._update_seed_in_layers(None)
        return out


    def fit(self, x, y=None, epochs=1, batch_size=32, shuffle=True, verbose=1):
        dtype = next(self.network.parameters()).dtype

        if not isinstance(x, DataLoader):
            x_tensor = torch.tensor(x, dtype=dtype)
            y_tensor = torch.tensor(y, dtype=dtype)
    
            dataset = TensorDataset(x_tensor, y_tensor)
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle)
        else:
            dataloader = x

        self.network.train()
        self.network = freeze_noMWE(self.network)

        begin_time = time.time()
        
        for epoch_idx in range(1, epochs + 1):
            # train and eval the model
            train_metrics = self.train_epoch(dataloader, epoch_idx, verbose)

            if verbose:
                print('Epoch {:03d} | Time {:5d}s | Train Loss {:.4f} | Weights {:.7f}'.format(
                      train_metrics['epoch_idx'],
                      int(time.time() - begin_time),
                      train_metrics['loss'],
                      train_metrics["weights"]), flush=True)
        
    
    def train_epoch(self, train_loader, epoch_idx, verbose):
        device = next(self.network.parameters()).device
        
        loss_avg = 0.0
        train_dataiter = iter(train_loader)

        for train_step in tqdm(range(1, len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=(verbose==0)):
            data, target = next(train_dataiter)
            data = data.to(device)
            target = target.to(device)

            # forward
            y_pred = self.network(data)
            loss = self.loss(y_pred, target)
            
            weights_dist = 0.
            count = 0.
            for w in filter(lambda p: p.requires_grad, self.network.parameters()):
                weights_dist += torch.sum(torch.abs(torch.nn.functional.softplus(w)))
                count += torch.sum(torch.ones_like(w))
                    
            weights_dist /= count
            weights_dist *= self.lambda_
            
            final_loss = loss - weights_dist

            # backward
            self.optimizer.zero_grad()
            final_loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                               self.clip_grad_norm)
            self.optimizer.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
                weights_avg = float(weights_dist)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg
        metrics['weights'] = weights_avg

        return metrics


    def _update_clip_in_layers(self, clip):
        for module in self.network.modules():
            if hasattr(module, "clip_"):
                setattr(module, "clip_", clip)


    def _update_seed_in_layers(self, seed):
        for module in self.network.modules():
            if hasattr(module, "seed_"):
                setattr(module, "seed_", seed)


    def fit_svd(self, x, batch_size=32):
        dataloader = self._x_to_dataloader(x, batch_size=batch_size)
        
        device = next(self.network.parameters()).device
        dummy = next(iter(dataloader)).to(device)
        self.network.eval()
        
        for module in self.network.modules():
            if hasattr(module, "fit_svd_"):
                module.fit_svd_ = "start"
        self.network(dummy)
        for batch in iter(dataloader):
            self.network(batch.to(device))
        for module in self.network.modules():
            if hasattr(module, "fit_svd_"):
                module.fit_svd_ = "end"
        self.network(dummy)


    def predict(self, x, batch_size=32, clip=None, seed=None):
        dataloader = self._x_to_dataloader(x, batch_size=batch_size)
        outputs = []
        for batch in iter(dataloader):
            out = self.__call__(batch, clip=clip, seed=seed)
            outputs.append(out)
        return torch.concat(outputs, axis=0).detach().numpy()


    def predict_mean(self, x, batch_size=32, clip=0., n_sample=1):
        dataloader = self._x_to_dataloader(x, batch_size=batch_size)
        preds = []
        kwargs = dict(batch_size=batch_size, clip=clip, seed=None)
        seeds = np.random.choice(10**9, size=n_sample, replace=True)
        for i in range(n_sample):
            kwargs["seed"] = int(seeds[i])
            preds.append(self.predict(dataloader, **kwargs))
        preds = np.stack(preds, axis=-1)
        pred_mean = np.mean(preds, axis=-1)
        return pred_mean


    def predict_std(self, x, batch_size=32, clip=None, n_sample=10):
        dataloader = self._x_to_dataloader(x, batch_size=batch_size)
        preds = []
        kwargs = dict(batch_size=batch_size, clip=clip, seed=None)
        seeds = np.random.choice(10**9, size=n_sample, replace=True)
        for i in range(n_sample):
            kwargs["seed"] = int(seeds[i])
            preds.append(self.predict(dataloader, **kwargs))
        preds = np.stack(preds, axis=-1)
        pred_std = np.std(preds, axis=-1)
        return pred_std


    def _x_to_dataloader(self, x, batch_size=32):
        dtype = next(self.network.parameters()).dtype
        if not isinstance(x, DataLoader):
            dataloader = DataLoader(torch.tensor(x, dtype=dtype),
                                    batch_size=batch_size,
                                    shuffle=False)
        else:
            dataloader = x
        return dataloader
