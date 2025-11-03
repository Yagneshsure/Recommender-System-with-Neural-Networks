import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.preprocessing.data_loader import load_processed_data, load_metadata
from src.models.gru4rec import GRU4Rec, create_dataloader
from src.models.baselines import PopularityRecommender, ItemKNNRecommender
from src.evaluation.metrics import evaluate_model, print_metrics, compare_models

def train_gru4rec(config):
    """Train GRU4Rec model."""
    print("="*60)
    print("Training GRU4Rec Model")
    print("="*60)
    
    # Load data
    train_data = load_processed_data(config, 'train')
    val_data = load_processed_data(config, 'val')
    metadata = load_metadata(config)
    n_items = metadata['n_items']
    
    print(f"\nDataset stats:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Number of items: {n_items}")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_data,
        config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = create_dataloader(
        val_data,
        config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Initialize model
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = GRU4Rec(
        n_items=n_items,
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/gru4rec_best.pt')
            print("  → Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('models/gru4rec_best.pt'))
    
    return model


def train_baselines(config):
    """Train baseline models."""
    print("\n" + "="*60)
    print("Training Baseline Models")
    print("="*60)
    
    train_data = load_processed_data(config, 'train')
    metadata = load_metadata(config)
    n_items = metadata['n_items']
    
    # Popularity baseline
    print("\n1. Training Popularity Recommender...")
    pop_model = PopularityRecommender()
    pop_model.fit(train_data)
    pop_model.save('models/popularity.pkl')
    
    # Item-KNN baseline
    print("\n2. Training Item-KNN Recommender...")
    knn_model = ItemKNNRecommender(k_neighbors=50)
    knn_model.fit(train_data, n_items)
    knn_model.save('models/item_knn.pkl')
    
    return pop_model, knn_model


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Train models based on argument
    if args.model == 'all' or args.model == 'baselines':
        pop_model, knn_model = train_baselines(config)
    
    if args.model == 'all' or args.model == 'gru4rec':
        gru_model = train_gru4rec(config)
    
    # Evaluate on test set
    if args.evaluate:
        print("\n" + "="*60)
        print("Evaluating Models on Test Set")
        print("="*60)
        
        test_data = load_processed_data(config, 'test')
        k_values = config['evaluation']['k_values']
        
        results_dict = {}
        
        # Evaluate baselines
        if args.model == 'all' or args.model == 'baselines':
            print("\nEvaluating Popularity...")
            pop_results = evaluate_model(pop_model, test_data, k_values, model_type='baseline')
            print_metrics(pop_results, "Popularity Baseline")
            results_dict['Popularity'] = pop_results
            
            print("\nEvaluating Item-KNN...")
            knn_results = evaluate_model(knn_model, test_data, k_values, model_type='baseline')
            print_metrics(knn_results, "Item-KNN Baseline")
            results_dict['Item-KNN'] = knn_results
        
        # Evaluate GRU4Rec
        if args.model == 'all' or args.model == 'gru4rec':
            print("\nEvaluating GRU4Rec...")
            gru_results = evaluate_model(gru_model, test_data, k_values, model_type='neural')
            print_metrics(gru_results, "GRU4Rec")
            results_dict['GRU4Rec'] = gru_results
        
        # Compare models
        if len(results_dict) > 1:
            compare_models(results_dict)
        
        # Save results
        with open('results/metrics.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print("Results saved to results/metrics.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train recommendation models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'baselines', 'gru4rec'],
                        help='Which model(s) to train')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate on test set after training')
    
    args = parser.parse_args()
    main(args)