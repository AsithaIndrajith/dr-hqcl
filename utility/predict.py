import torch as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_accuracy( data_loder, device, model ):
    # Make predictions
    y_pred = []
    y_pred_2 = []
    test_accuracy = 0
    with nn.no_grad():
        for inputs, labels in data_loder:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            y_pred_2.extend(outputs)
            _, predicted_labels = nn.max(outputs, dim=1)
            y_pred.extend(predicted_labels.tolist())
            test_accuracy += (predicted_labels == labels).sum().item()  # Count correctly predicted samples

    test_accuracy = 100.0 * test_accuracy / len(data_loder.dataset)
    return test_accuracy, y_pred, y_pred_2

def train_model(model, lossFunction, optimizer, scheduler, train_loader, test_loader, num_epochs, validation=True, regularize=True):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Set the model to training mode
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Assuming you have a data loader for training
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossFunction(outputs, labels)
            loss.backward()
            if regularize:
                scheduler.step()
            else:
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        average_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / train_total
        train_loss_history.append(average_train_loss)
        train_acc_history.append(train_accuracy)

        log = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"

        if validation :
            # Evaluate on the validation set
            model.eval()  # Set the model to evaluation mode

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:  # Assuming you have a data loader for validation
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = lossFunction(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            average_val_loss = val_loss / len(test_loader.dataset)
            val_accuracy = 100.0 * val_correct / val_total
            val_loss_history.append(average_val_loss)
            val_acc_history.append(val_accuracy)

            log = str(log) + str(f", Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}")

        # Print training loss for each epoch
        print(log)

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history