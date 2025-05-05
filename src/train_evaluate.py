import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler



def calc_loss_batch(predictions, targets):
    # Use a PyTorch loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # Calculate loss (this returns a tensor that can be used with backward())
    loss = criterion(predictions, targets)
    # Return the loss tensor and number of samples
    return loss, targets.size(0)




def setup_distributed(rank, world_size):
    """
    Initialize the distributed environment.

    Args:
        rank: Unique ID of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set the device for this process
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def train_classifier_ddp(local_rank, world_size, model, train_dataset, val_dataset,
                         optimizer_fn, num_epochs, batch_size=32, log_interval=100,
                         scheduler_fn=None, use_mixed_precision=True, num_workers=4):
    """
    Train a classifier model with DistributedDataParallel.

    Args:
        local_rank: Local rank of this process
        world_size: Total number of processes
        model: PyTorch model (not wrapped with DDP yet)
        train_dataset: Dataset for training
        val_dataset: Dataset for validation
        optimizer_fn: Function that returns optimizer when given model parameters
        num_epochs: Number of training epochs
        batch_size: Batch size per GPU
        log_interval: Number of batches between logging updates
        scheduler_fn: Function that returns scheduler when given optimizer
        use_mixed_precision: Whether to use mixed precision training
        num_workers: Number of data loading workers per GPU

    Returns:
        dict: Training history including losses and accuracies (only on rank 0)
    """
    # Set up the distributed environment
    setup_distributed(local_rank, world_size)

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True
    )

    # Set device
    device = torch.device(f"cuda:{local_rank}")

    # Move model to device
    model = model.to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create optimizer and scheduler
    optimizer = optimizer_fn(model.parameters())
    scheduler = scheduler_fn(optimizer) if scheduler_fn else None

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if use_mixed_precision and torch.cuda.is_available() else None

    # Initialize history dictionary (only on rank 0)
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    } if local_rank == 0 else None

    for epoch in range(num_epochs):
        # Set the epoch for the train sampler
        train_sampler.set_epoch(epoch)

        if local_rank == 0:
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

            # Move data to device (with non_blocking for better performance)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass with mixed precision
            optimizer.zero_grad()

            if use_mixed_precision and torch.cuda.is_available():
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

            # Print batch statistics at regular intervals (only on rank 0)
            if batch_idx % log_interval == 0 and local_rank == 0:
                batch_accuracy = 100 * batch_correct / batch_total
                avg_batch_loss = batch_loss / batch_total

                print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_dataloader)}: "
                      f"Loss: {avg_batch_loss:.4f}, "
                      f"Accuracy: {batch_accuracy:.2f}%")

                # Reset batch statistics
                batch_loss = 0
                batch_correct = 0
                batch_total = 0

        # Gather metrics from all processes
        train_loss_tensor = torch.tensor(sum(train_losses), device=device)
        train_total_tensor = torch.tensor(train_total, device=device)
        train_correct_tensor = torch.tensor(train_correct, device=device)

        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)

        # Calculate training metrics for the whole epoch
        epoch_train_loss = train_loss_tensor.item() / train_total_tensor.item()
        epoch_train_accuracy = 100 * train_correct_tensor.item() / train_total_tensor.item()

        # Validation phase
        model.eval()
        val_losses = []
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch

                # Move data to device
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Forward pass
                predictions = model(x)[:, -1, :]
                loss, num_samples = calc_loss_batch(predictions, y)

                # Calculate accuracy
                _, predicted = torch.max(predictions.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

                # Record loss statistics
                val_losses.append(loss.item() * num_samples)

        # Gather validation metrics from all processes
        val_loss_tensor = torch.tensor(sum(val_losses), device=device)
        val_total_tensor = torch.tensor(val_total, device=device)
        val_correct_tensor = torch.tensor(val_correct, device=device)

        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)

        # Calculate validation metrics
        epoch_val_loss = val_loss_tensor.item() / val_total_tensor.item()
        epoch_val_accuracy = 100 * val_correct_tensor.item() / val_total_tensor.item()

        # Step the learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # Store metrics in history (only on rank 0)
        if local_rank == 0:
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_accuracy)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_accuracy)

            # Print epoch results
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train loss: {epoch_train_loss:.4f}, Train accuracy: {epoch_train_accuracy:.2f}%")
            print(f"Val loss: {epoch_val_loss:.4f}, Val accuracy: {epoch_val_accuracy:.2f}%")
            if use_mixed_precision and torch.cuda.is_available():
                print("Training with mixed precision (FP16/FP32)")
            print(f"Training with DistributedDataParallel on {world_size} GPUs")

    # Clean up
    cleanup_distributed()

    return history