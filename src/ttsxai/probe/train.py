import torch
import torch.nn as nn
from torch.autograd import Variable


def train_probe(
    probe,
    X_train,
    y_train,
    task_type,
    lambda_l1=0,
    lambda_l2=0,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
    use_gpu=True
):
    print(f"Training {task_type} probe")

    if lambda_l1 is None or lambda_l2 is None:
        raise ValueError("Regularization weights cannot be None")

    if use_gpu:
        probe = probe.cuda()

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid `task_type`")

    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    for epoch in range(num_epochs):
        num_tokens = 0
        avg_loss = 0
        for inputs, labels in progressbar(
            utils.batch_generator(X_tensor, y_tensor, batch_size=batch_size),
            desc="epoch [%d/%d]" % (epoch + 1, num_epochs),
        ):
            num_tokens += inputs.shape[0]
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs = inputs.float()
            inputs = Variable(inputs)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            outputs = probe(inputs)
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

            avg_loss += loss.item()

        print(
            "Epoch: [%d/%d], Loss: %.4f"
            % (epoch + 1, num_epochs, avg_loss / num_tokens)
        )