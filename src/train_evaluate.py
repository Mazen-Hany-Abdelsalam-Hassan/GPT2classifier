import torch
from torch.cuda.amp import autocast, GradScaler


def calc_loss_batch(predictions, targets):
    # Use a PyTorch loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # Calculate loss (this returns a tensor that can be used with backward())
    loss = criterion(predictions, targets)
    # Return the loss tensor and number of samples
    return loss, targets.size(0)


def train_classifier(model, train_dataloader, validation_dataloader,
                     optimizer, num_epochs, device, log_interval=100, scheduler=None,
                     use_mixed_precision=True):
    """
    Train a classifier model with regular batch interval logging.

    Args:
        model: PyTorch model
        train_dataloader: DataLoader for training data
        validation_dataloader: DataLoader for validation data
        optimizer: PyTorch optimizer
        num_epochs: Number of training epochs
        device: Device to use for training ('cuda' or 'cpu')
        log_interval: Number of batches between logging updates
        scheduler: Optional learning rate scheduler
        use_mixed_precision: Whether to use mixed precision training (FP16)

    Returns:
        dict: Training history including losses and accuracies
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if use_mixed_precision and device == torch.device("cuda") else None

    # Check if mixed precision is available
    if use_mixed_precision and device != torch.device("cuda"):
        print("Mixed precision training requires CUDA. Falling back to full precision.")
        use_mixed_precision = False

    # Move model to device
    model = model.to(device)

    for epoch in range(num_epochs):
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_losses = []
        train_total = 0
        train_correct = 0
        batch_loss = 0
        batch_correct = 0
        batch_total = 0

        for batch_idx, batch in enumerate(train_dataloader, 1):
            x, y = batch

            # Move data to device
            x = x.to(device)
            y = y.to(device)

            # Forward pass with mixed precision
            optimizer.zero_grad()

            if use_mixed_precision:
                # Use autocast for mixed precision forward pass
                with autocast():
                    predictions = model(x)[:, -1, :]
                    loss, num_samples = calc_loss_batch(predictions, y)

                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard full precision forward and backward pass
                predictions = model(x)[:, -1, :]
                loss, num_samples = calc_loss_batch(predictions, y)
                loss.backward()
                optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            current_correct = (predicted == y).sum().item()

            # Update counters
            train_total += y.size(0)
            train_correct += current_correct
            batch_total += y.size(0)
            batch_correct += current_correct

            # Record loss statistics
            current_loss = loss.item() * num_samples
            train_losses.append(current_loss)
            batch_loss += current_loss

            # Print batch statistics at regular intervals
            if batch_idx % log_interval == 0:
                batch_accuracy = 100 * batch_correct / batch_total
                avg_batch_loss = batch_loss / batch_total

                print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_dataloader)}: "
                      f"Loss: {avg_batch_loss:.4f}, "
                      f"Accuracy: {batch_accuracy:.2f}%")

                # Reset batch statistics
                batch_loss = 0
                batch_correct = 0
                batch_total = 0

        # Calculate training metrics for the whole epoch
        epoch_train_loss = sum(train_losses) / train_total
        epoch_train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_losses = []
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for batch in validation_dataloader:
                x, y = batch

                # Move data to device
                x = x.to(device)
                y = y.to(device)

                # Forward pass (no need for autocast in eval as we want consistent results)
                predictions = model(x)[:, -1, :]
                loss, num_samples = calc_loss_batch(predictions, y)

                # Calculate accuracy
                _, predicted = torch.max(predictions.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

                # Record loss statistics
                val_losses.append(loss.item() * num_samples)

        # Calculate validation metrics
        epoch_val_loss = sum(val_losses) / val_total
        epoch_val_accuracy = 100 * val_correct / val_total

        # Step the learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Store metrics in history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_accuracy)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_accuracy)

        # Print epoch results
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train loss: {epoch_train_loss:.4f}, Train accuracy: {epoch_train_accuracy:.2f}%")
        print(f"Val loss: {epoch_val_loss:.4f}, Val accuracy: {epoch_val_accuracy:.2f}%")
        if use_mixed_precision:
            print("Training with mixed precision (FP16/FP32)")

    return history


def evaluate_classifier(model, dataloader, device, use_mixed_precision=False):
    """
    Evaluate a classifier model on a dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to use for evaluation ('cuda' or 'cpu')
        use_mixed_precision: Whether to use mixed precision for evaluation

    Returns:
        dict: Evaluation metrics including loss and accuracy
    """
    # Make sure model is in evaluation mode
    model.eval()
    model = model.to(device)

    # Initialize counters
    total_loss = 0
    total_samples = 0
    total_correct = 0

    # Disable gradient computation during evaluation
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch

            # Move data to device
            x = x.to(device)
            y = y.to(device)

            # Forward pass with optional mixed precision
            if use_mixed_precision and device == torch.device("cuda"):
                with autocast():
                    predictions = model(x)[:, -1, :]
                    loss_result = calc_loss_batch(predictions, y)
            else:
                predictions = model(x)[:, -1, :]
                loss_result = calc_loss_batch(predictions, y)

            # Handle different return types from calc_loss_batch
            if isinstance(loss_result, tuple):
                loss, num_samples = loss_result
            else:
                # If only a single value is returned, assume it's the loss value
                loss = loss_result
                num_samples = y.size(0)

            # Get the scalar value if it's a tensor
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            current_correct = (predicted == y).sum().item()

            # Update counters
            total_samples += y.size(0)
            total_correct += current_correct
            total_loss += loss_value * num_samples

    # Calculate metrics
    avg_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples

    # Return metrics as a dictionary
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy
    }

    return metrics