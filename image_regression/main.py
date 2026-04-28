"""
=============================================================
  Image Feature-Based Prediction using Linear Regression
=============================================================
  Author   : Vidyanand Yadav & Ayush Maurya
  Libraries: NumPy, Pandas, Matplotlib, OpenCV, Scikit-learn
  Purpose  : Extract image features and predict a target value
             (simulated "price") using Linear Regression.
=============================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ─────────────────────────────────────────────
# STEP 1: Load images from the "images" folder
# ─────────────────────────────────────────────
def load_images(folder_path):
    """Load all images from the given folder. Returns list of (filename, image) tuples."""
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    images = []

    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder '{folder_path}' not found. Please run generate_sample_images.py first.")
        return images

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img))

    print(f"[INFO] Loaded {len(images)} image(s) from '{folder_path}'")
    return images


# ─────────────────────────────────────────────
# STEP 2: Extract features from each image
# ─────────────────────────────────────────────
def extract_features(images):
    """
    For each image:
      - Convert to grayscale
      - Compute average brightness (mean pixel value)
      - Count edges using Canny edge detection
    Returns a Pandas DataFrame with columns: filename, brightness, edges
    """
    records = []

    for filename, img in images:
        # Convert BGR image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Feature 1: Average brightness (mean of all pixel values, 0–255)
        brightness = np.mean(gray)

        # Feature 2: Edge count using Canny edge detector
        edges_img = cv2.Canny(gray, threshold1=100, threshold2=200)
        edge_count = np.sum(edges_img > 0)  # count non-zero (edge) pixels

        records.append({
            'filename'  : filename,
            'brightness': round(brightness, 4),
            'edges'     : edge_count
        })

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────
# STEP 3: Create target variable "price"
# ─────────────────────────────────────────────
def create_target(df):
    """
    Simulate a target variable 'price' using a weighted formula:
        price = brightness * 0.5 + edges * 0.01
    Noise is added to simulate real-world unpredictability.
    """
    df['price'] = (df['brightness'] * 0.5) + (df['edges'] * 0.01)
    np.random.seed(42)
    noise = np.random.normal(0, 5, size=len(df))
    df['price'] = (df['price'] + noise).round(4)
    return df


# ─────────────────────────────────────────────
# STEP 4: Train Linear Regression model
# ─────────────────────────────────────────────
def train_model(df):
    """
    Train a Linear Regression model using brightness and edges as features,
    and price as the target variable.
    Returns the trained model, test sets, and predictions.
    """
    X = df[['brightness', 'edges']].values
    y = df['price'].values

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")
    print(f"       Coefficients : brightness={model.coef_[0]:.4f}, edges={model.coef_[1]:.4f}")
    print(f"       Intercept    : {model.intercept_:.4f}")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"       MSE (test)   : {mse:.4f}")
    print(f"       R² Score     : {r2:.4f}")

    return model, X_test, y_test, y_pred


# ─────────────────────────────────────────────
# STEP 5: Add predictions to the full dataset
# ─────────────────────────────────────────────
def add_predictions(df, model):
    """Use the trained model to predict price for all rows and add as a new column."""
    X_all = df[['brightness', 'edges']].values
    df['predicted_price'] = model.predict(X_all).round(4)
    return df


# ─────────────────────────────────────────────
# STEP 6: Visualizations
# ─────────────────────────────────────────────
def plot_results(df, y_test, y_pred):
    """Generate two plots: Actual vs Predicted, and Brightness vs Price."""

    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Image Feature-Based Prediction using Linear Regression", fontsize=14)

    # --- Plot 1: Actual vs Predicted values ---
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, color='steelblue', edgecolors='black', s=80, label='Predictions')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Ideal fit')
    ax1.set_xlabel("Actual Price")
    ax1.set_ylabel("Predicted Price")
    ax1.set_title("Actual vs Predicted Price")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: Brightness vs Price ---
    ax2 = axes[1]
    ax2.scatter(df['brightness'], df['price'], color='darkorange', edgecolors='black',
                s=80, label='Actual Price')
    ax2.scatter(df['brightness'], df['predicted_price'], color='green', marker='^',
                s=80, label='Predicted Price')
    ax2.set_xlabel("Brightness (mean pixel intensity)")
    ax2.set_ylabel("Price")
    ax2.set_title("Brightness vs Price")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- Metrics bar below both plots ---
    metrics_text = (
        f"  R² Score : {r2:.4f}     |     "
        f"Test Samples : {len(y_test)}     |     "
        f"Total Images : {len(df)}"
    )
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=10,
             color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a237e', edgecolor='#3949ab'))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("output_plots.png", dpi=150)
    print("[INFO] Plots saved as 'output_plots.png'")
    plt.show()


# ─────────────────────────────────────────────
# STEP 7: Show sample image before/after grayscale
# ─────────────────────────────────────────────
def show_sample_image(images):
    """Display the first image in color and grayscale side by side."""
    if not images:
        return

    filename, img = images[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Sample Image: {filename}", fontsize=13)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (Color)")
    axes[0].axis('off')

    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title("After Grayscale Conversion")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("sample_image_comparison.png", dpi=150)
    print("[INFO] Sample image comparison saved as 'sample_image_comparison.png'")
    plt.show()



# MAIN — ties everything together

def main():
    print("=" * 60)
    print("  Image Feature-Based Prediction using Linear Regression")
    print("=" * 60)

    # 1. Load images from the images/ folder
    images = load_images("images")
    if not images:
        print("[ERROR] No images found. Please run generate_sample_images.py first.")
        return

    # 2. Show sample image (before/after grayscale)
    show_sample_image(images)

    # 3. Extract features into a DataFrame
    df = extract_features(images)

    # 4. Create target variable
    df = create_target(df)

    # 5. Print dataset preview
    print("\n--- Dataset Preview ---")
    print(df.to_string(index=False))

    # 6. Train model
    print("\n--- Model Training ---")
    model, X_test, y_test, y_pred = train_model(df)

    # 7. Add predictions to dataset
    df = add_predictions(df, model)

    # 8. Print prediction results
    print("\n--- Prediction Results ---")
    print(df[['filename', 'brightness', 'edges', 'price', 'predicted_price']].to_string(index=False))

    # 9. Save dataset to CSV
    csv_path = "image_features_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Dataset saved as '{csv_path}'")

    # 10. Plot results
    print("\n--- Generating Plots ---")
    plot_results(df, y_test, y_pred)

    print("\n[DONE] Project execution complete.")


if __name__ == "__main__":
    main()
