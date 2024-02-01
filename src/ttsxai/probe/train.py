"""
    adapted from:
    * https://github.com/fdalvi/NeuroX/blob/master/neurox/interpretation/linear_probe.py
"""
import os
import time

import torch
import torch.nn as nn


def cycle(dl):
    while True:
        for data in dl:
            yield data


def move_to_device(batch, device):
    """Move the given batch to the specified device."""
    if isinstance(batch, (list, tuple)):
        return [move_to_device(x, device) for x in batch]
    return batch.to(device)


def train_probe(
    probe,
    train_dataset,
    val_dataset,
    task_type,
    lambda_l1=0,
    lambda_l2=0,
    num_epochs=60,
    num_train_steps_per_epoch=500,
    batch_size=256,
    learning_rate=0.001,
    use_gpu=True,
    logger=None,
    logdir=None,
):
    print(f"Training {task_type} probe")

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    probe.to(device)

    if lambda_l1 is None or lambda_l2 is None:
        raise ValueError("Regularization weights cannot be None")

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid `task_type`")

    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    train_dataloader = cycle(torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True
    ))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True
    )

    time_dict = {}
    total_start_time = time.time()
    best_val_loss = float('inf')
    for epoch in range(0, num_epochs):
        epoch_start_time = time.time()

        # === Train === 
        training_start_time = time.time()
        probe.train()
        train_loss = 0
        for step in range(num_train_steps_per_epoch):
            batch = next(train_dataloader)
            activations, labels = move_to_device(batch, device)
            outputs = probe(activations)
            if task_type == "regression":
                outputs = outputs.squeeze()
            weights = list(probe.parameters())[0]
            loss = (
                criterion(outputs, labels)
                + lambda_l1 * l1_penalty(weights)
                + lambda_l2 * l2_penalty(weights)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() 

        train_loss = train_loss / num_train_steps_per_epoch
        time_dict['time/training (s)'] = time.time() - training_start_time

        # === Validation === 
        validation_start_time = time.time()
        probe.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                activations, labels = move_to_device(batch, device)
                outputs = probe(activations)
                if task_type == "regression":
                    outputs = outputs.squeeze()
                loss = (
                    criterion(outputs, labels)
                )
                val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)
        time_dict['time/validation (s)'] = time.time() - validation_start_time

        # === Save === 
        save_start_time = time.time()

        steps = (epoch + 1) * num_train_steps_per_epoch
        total_steps = num_epochs * num_train_steps_per_epoch
        # save at most 10 snapshot!
        label = steps // int(total_steps * 0.1) * int(total_steps * 0.1)
        snapshot = {
            'step': step,
            'model': probe.state_dict()
        }
        savepath = os.path.join(logdir, f'state_{label}.pt')
        torch.save(snapshot, savepath)
        # save best model
        if val_loss < best_val_loss: 
            snapshot = {
                'step': step,
                'model': probe.state_dict()
            }
            savepath = os.path.join(logdir, f'state_best.pt')
            torch.save(snapshot, savepath)
            best_val_loss = val_loss

        time_dict['time/save (s)'] = time.time() - save_start_time
        time_dict['time/epoch (s)'] = time.time() - epoch_start_time
        time_dict['time/total (s)'] = time.time() - total_start_time

        # === Logging === 
        logger.record_tabular('train/Loss', train_loss)
        logger.record_tabular('val/Loss', val_loss)
        logger.record_tabular('val/Best Loss', best_val_loss)
        logger.record_dict(time_dict)
        logger.record_tabular('num train steps total', steps)
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)


def l1_penalty(var):
    """
    L1/Lasso regularization penalty

    Parameters
    ----------
    var : torch.Variable
        Torch variable representing the weight matrix over which the penalty
        should be computed

    Returns
    -------
    penalty : torch.Variable
        Torch variable containing the penalty as a single floating point value

    """
    return torch.abs(var).sum()


def l2_penalty(var):
    """
    L2/Ridge regularization penalty.

    Parameters
    ----------
    var : torch.Variable
        Torch variable representing the weight matrix over which the penalty
        should be computed

    Returns
    -------
    penalty : torch.Variable
        Torch variable containing the penalty as a single floating point value

    Notes
    -----
    The penalty is derived from the L2-norm, which has a square root. The exact
    optimization can also be done without the square root, but this makes no
    difference in the actual output of the optimization because of the scaling
    factor used along with the penalty.

    """
    return torch.sqrt(torch.pow(var, 2).sum())