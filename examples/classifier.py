import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score
import time


class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return output


def count_class_samples(y):
    class_counts = torch.bincount(y)
    print(f"Class distribution in the dataset: {class_counts}")
    return class_counts


def count_true_predictions(y_true, y_pred, num_classes):
    correct_predictions = (y_true == y_pred)
    true_class_preds = torch.zeros(num_classes)

    for i in range(num_classes):
        true_class_preds[i] = (y_true[correct_predictions] == i).sum().item()

    print(f"Correct predictions per class: {true_class_preds}")
    return true_class_preds


def train(model, X_train, y_train, X_val, y_val, num_epochs=50, learning_rate=0.001, weight_decay=1e-5, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metrics = {
        'training_losses': [],
        'validation_losses': [],
        'val_accuracies': [],
        'val_recalls': [],
        'val_precisions': [],
        'val_f1_scores': [],
        'epoch_times': []
    }

    start_time = time.time()
    early_stop_counter = 0
    best_f1_score = -float('inf')
    best_val_loss = float('inf')

    X_train = torch.Tensor(X_train) if not isinstance(X_train, torch.Tensor) else X_train
    y_train = torch.LongTensor(y_train) if not isinstance(y_train, torch.Tensor) else y_train
    X_val = torch.Tensor(X_val) if not isinstance(X_val, torch.Tensor) else X_val
    y_val = torch.LongTensor(y_val) if not isinstance(y_val, torch.Tensor) else y_val

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        epoch_duration = time.time() - epoch_start
        metrics['epoch_times'].append(epoch_duration)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            _, val_preds = torch.max(val_outputs, 1)

        acc = accuracy_score(y_val.cpu(), val_preds.cpu())
        recall = recall_score(y_val.cpu(), val_preds.cpu(), average='macro', zero_division=0)
        precision = precision_score(y_val.cpu(), val_preds.cpu(), average='macro', zero_division=0)
        f1 = f1_score(y_val.cpu(), val_preds.cpu(), average='macro')

        metrics['training_losses'].append(loss.item())
        metrics['validation_losses'].append(val_loss)
        metrics['val_accuracies'].append(acc)
        metrics['val_recalls'].append(recall)
        metrics['val_precisions'].append(precision)
        metrics['val_f1_scores'].append(f1)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, "
              f"Accuracy: {acc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, "
              f"Time: {epoch_duration:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    total_duration = time.time() - start_time

    _, train_preds = torch.max(outputs, 1)
    train_accuracy = accuracy_score(y_train.cpu(), train_preds.cpu())
    print(f"\nFinal Training Accuracy: {train_accuracy:.4f}")
    print(f"Total Training Time: {total_duration:.2f} seconds")
    print(f"Model converged in {epoch + 1} epochs.")

    print("Training data class distribution:")
    count_class_samples(y_train)

    print("Correctly predicted class distribution for training data:")
    count_true_predictions(y_train, train_preds, num_classes=model.fc2.out_features)

    return model, metrics


def evaluate_model(model, X_test, y_test, multi_class='ovr'):
    metrics = {
        'loss': [],
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1_score': [],
        'roc_auc': []
    }

    X_test = torch.Tensor(X_test) if not isinstance(X_test, torch.Tensor) else X_test
    y_test = torch.LongTensor(y_test) if not isinstance(y_test, torch.Tensor) else y_test

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        probabilities = torch.softmax(test_outputs, dim=1).cpu().numpy()
        _, test_preds = torch.max(test_outputs, 1)

    acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    recall = recall_score(y_test.cpu(), test_preds.cpu(), average='macro', zero_division=0)
    precision = precision_score(y_test.cpu(), test_preds.cpu(), average='macro', zero_division=0)
    f1 = f1_score(y_test.cpu(), test_preds.cpu(), average='macro')

    try:
        roc_auc = roc_auc_score(y_test.cpu().numpy(), probabilities[:, 1])
    except ValueError as e:
        print(f"ROC AUC calculation error: {e}")
        roc_auc = None

    metrics['accuracy'].append(acc)
    metrics['recall'].append(recall)
    metrics['precision'].append(precision)
    metrics['f1_score'].append(f1)
    metrics['roc_auc'].append(roc_auc)

    print(f"\nTest Set Evaluation:")
    print(f"Accuracy: {acc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")

    print("Test data class distribution:")
    count_class_samples(y_test)

    print("Correctly predicted class distribution for test data:")
    count_true_predictions(y_test, test_preds, num_classes=model.fc2.out_features)

    return metrics
