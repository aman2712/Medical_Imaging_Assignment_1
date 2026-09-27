import torch


def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    criterion,
    number_of_epochs,
    learning_rate=0.001
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    training_losses = []
    validation_losses = []

    for epoch in range(number_of_epochs):
        model.train()

        train_logits = model(x_train)
        train_loss = criterion(train_logits, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_logits = model(x_val)
            validation_loss = criterion(validation_logits, y_val)

        training_losses.append(train_loss.item())
        validation_losses.append(validation_loss.item())

    return model, training_losses, validation_losses


def train_cnn(model, train_loader, val_loader, criterion, optimizer, epochs):
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0

        for images, labels in train_loader:
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)

        model.eval()
        validation_loss_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
                validation_loss_total += loss.item() * images.size(0)

        train_loss = train_loss_total / len(train_loader.dataset)
        validation_loss = validation_loss_total / len(val_loader.dataset)

        training_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- train loss: {train_loss:.4f} "
            f"- validation loss: {validation_loss:.4f}"
        )

    return training_losses, validation_losses