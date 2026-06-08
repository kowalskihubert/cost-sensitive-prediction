import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    # Load limited features just to speed up the basic debug, or all if it's small.
    # The dataset size is 5000 x 500. This easily fits in memory.
    # Skip header row (contains column names like 'V1', 'V2', etc.)
    X_train = np.loadtxt('../data/x_train.txt', skiprows=1)
    y_train = np.loadtxt('../data/y_train.txt', skiprows=1)
    X_test = np.loadtxt('../data/x_test.txt', skiprows=1)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    models = {
        'Random Forest (max_depth=5)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'Logistic Regression (L2)': make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    }

    plt.figure(figsize=(18, 10))
    
    all_test_preds = []
    
    for i, (name, model) in enumerate(models.items()):
        print(f"\nEvaluating {name}...")
        
        # 1. Get Out-Of-Fold predictions on Train to know what the reference 50/50 distribution looks like
        print("  Computing OOF on Train...")
        train_oof_preds = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
        
        # 2. Train on full TRAIN and predict on TEST
        print("  Fitting on full Train and predicting on Test...")
        model.fit(X_train, y_train)
        test_preds = model.predict_proba(X_test)[:, 1]
        all_test_preds.append(test_preds)
        
        # Stats
        print(f"  Train OOF Mean Prob: {train_oof_preds.mean():.4f}")
        print(f"  Test Mean Prob:      {test_preds.mean():.4f}")
        
        # Plot 1: Histogram to see if the entire distribution shifted left
        plt.subplot(2, 3, i*2 + 1)
        plt.hist(train_oof_preds, bins=50, alpha=0.5, density=True, label='Train OOF (50% Actual True)')
        plt.hist(test_preds, bins=50, alpha=0.5, density=True, label='Test Set (Unknown True)')
        plt.title(f'{name}:\nProbability Distributions (Histogram)')
        plt.xlabel('Predicted Probability (Class 1)')
        plt.ylabel('Density')
        plt.legend()
        
        # Plot 2: Sorted probabilities (Waterfall) to look for the "sudden drop"
        plt.subplot(2, 3, i*2 + 2)
        plt.plot(np.sort(train_oof_preds)[::-1], label='Train OOF', linewidth=2)
        plt.plot(np.sort(test_preds)[::-1], label='Test Set', linewidth=2)
        plt.title(f'{name}:\nSorted Probabilities (Waterfall)')
        plt.xlabel('Customer Rank (Sorted by Prob)')
        plt.ylabel('Predicted Probability')
        
        # Highlight markers at 500 and 2500
        plt.axvline(x=500, color='r', linestyle='--', alpha=0.7, label='Rank = 500')
        plt.axvline(x=2500, color='g', linestyle='--', alpha=0.7, label='Rank = 2500')
        
        plt.grid(alpha=0.3)
        plt.legend()
    
    # Plot 3: Ensemble average of all test predictions sorted
    print("\nComputing ensemble average of Test predictions...")
    ensemble_test_preds = np.mean(all_test_preds, axis=0)
    sorted_ensemble_preds = np.sort(ensemble_test_preds)[::-1]
    
    plt.subplot(2, 3, 5)
    plt.plot(sorted_ensemble_preds, linewidth=2.5, color='darkblue', label='Ensemble Avg (Sorted)')
    
    # Compute derivative to detect sudden drops
    derivative = np.diff(sorted_ensemble_preds)
    threshold_drop = np.percentile(np.abs(derivative), 90)
    drops = np.where(np.abs(derivative) > threshold_drop)[0]
    
    if len(drops) > 0:
        for drop_idx in drops[:3]:  # Show top 3 drops
            plt.axvline(x=drop_idx, color='red', linestyle=':', alpha=0.6)
            plt.text(drop_idx, sorted_ensemble_preds[0] * 0.95, f'Drop at {drop_idx}', rotation=90, fontsize=9)
    
    plt.axvline(x=500, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Rank = 500')
    plt.axvline(x=2500, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Rank = 2500')
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Break-even (~0.5)')
    
    plt.title('Ensemble Average: Sorted Test Predictions\n(Looking for sudden drops)')
    plt.xlabel('Customer Rank (Sorted by Prob)')
    plt.ylabel('Predicted Probability')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('test_prior_analysis.png', format='png', dpi=300)
    print("\nVisualizations saved to 'debug/test_prior_analysis.png'")
    print("Done. Check the bottom-right plot for sudden drops.")

if __name__ == '__main__':
    main()