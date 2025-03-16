if __name__ == '__main__':
    import optuna
    import logging
    import joblib
    import pickle
    import gc
    import torch
    from tpot import TPOTClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torchvision.datasets import FashionMNIST
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.optim as optim

    # Free up unused memory before running TPOT
    gc.collect()
    torch.cuda.empty_cache()

    # Configure logging
    logging.basicConfig(filename="tpot_progress.log", level=logging.INFO, format="%(asctime)s - %(message)s")

    # Load and preprocess the dataset
    def preprocess_data(dataset):
        data = dataset.data.numpy().reshape(len(dataset), -1)
        scaler = StandardScaler()
        return scaler.fit_transform(data), dataset.targets.numpy()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)
    
    X_train, y_train = preprocess_data(train_dataset)
    X_train, y_train = X_train[:10000], y_train[:10000]  # Use only 10,000 samples to reduce memory usage
    X_test, y_test = preprocess_data(test_dataset)

    # Model Selection using AutoML
    def train_tpot():
        print("Model Selection using AutoML started.")
        logging.info("Starting TPOT model training.")

        tpot = TPOTClassifier(
            generations=2,  # Reduce number of generations (default is high)
            population_size=5,  # Fewer models per generation (default is 100)
            n_jobs=1,  # Disable parallel processing to prevent memory overload
            early_stop=1,  # Stop if no improvement after 1 generation
            warm_start=False,  # Retains previous progress and prevents recomputation
            periodic_checkpoint_folder="tpot_checkpoints"  # Save progress periodically
        )

        print("Training TPOT...")  # ✅ Removed unnecessary loop
        tpot.fit(X_train, y_train)  

        print("Best Pipeline Found!")
        logging.info("Best pipeline found.")

        best_pipeline = tpot.fitted_pipeline_

        joblib.dump(best_pipeline, 'best_tpot_model.pkl')
        tpot.export('best_model.py')

        accuracy = accuracy_score(y_test, tpot.predict(X_test))
        print(f"Best Model Accuracy: {accuracy:.4f}")
        logging.info(f"Best Model Accuracy: {accuracy:.4f}")

        return tpot, best_pipeline, accuracy

    best_model, best_pipeline, best_accuracy = train_tpot()

    # Hyperparameter Optimization
    def optimize_hyperparams(trial):
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # ✅ Fix DataLoader to avoid high memory usage
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                           torch.tensor(y_train, dtype=torch.long)), 
            batch_size=batch_size, 
            shuffle=True
        )

        model.train()
        for epoch in range(5):  
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate on test data
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predictions = torch.argmax(test_outputs, dim=1)
            accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())

        return accuracy  # ✅ Optuna will maximize this

    # ✅ Reduce Optuna trials from 10 to 5 for faster execution
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_hyperparams, n_trials=5)  
    print("Best hyperparameters:", study.best_params)

    # Log AutoML Results
    automl_results = {
        "best_model_pipeline": best_pipeline,
        "best_model_accuracy": best_accuracy,
        "best_hyperparameters": study.best_params
    }
    
    import json
    with open("automl_results.json", "w") as f:
        json.dump(automl_results, f, indent=4)  # Save results in a human-readable format

    # Final Model Evaluation
    y_pred = best_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
