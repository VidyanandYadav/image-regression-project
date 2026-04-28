# Project Report

## Image Feature-Based Prediction using Linear Regression

---

| | |
|---|---|
| **Subject** | Machine Learning / Computer Vision |
| **Topic** | Image Feature Extraction and Regression |
| **Language** | Python 3 |
| **Date** | April 2026 |

---

## 1. Introduction

In this project, we explore how visual features extracted from images can be used to predict a numerical value using a supervised machine learning algorithm — **Linear Regression**.

The project simulates a real-world scenario where image characteristics (such as brightness and structural complexity) are mapped to a target value called **"price"**. This concept is directly applicable to domains like real estate (predicting property prices from photos), e-commerce (product image quality scoring), and medical imaging (severity scoring from scans).

Since collecting a real labeled image-price dataset requires significant resources, we simulate the target variable using a deterministic formula — a standard and accepted academic approach for demonstrating a complete ML pipeline.

---

## 2. Objectives

- Load and preprocess images using OpenCV
- Extract meaningful numerical features from raw image data
- Build and train a Linear Regression model using Scikit-learn
- Evaluate model performance using standard metrics (MSE, R²)
- Visualize results through plots
- Export the final dataset with predictions to a CSV file

---

## 3. Technologies Used

| Library | Version | Purpose |
|---|---|---|
| Python | 3.x | Core programming language |
| NumPy | latest | Numerical operations on pixel arrays |
| Pandas | latest | Dataset creation and CSV export |
| Matplotlib | latest | Plotting graphs and displaying images |
| OpenCV (cv2) | latest | Image loading, grayscale conversion, edge detection |
| Scikit-learn | latest | Linear Regression model, train/test split, metrics |

---

## 4. Project Structure

```
project/
│
├── image_regression/
│   ├── main.py                      # Main pipeline code
│   ├── generate_sample_images.py    # Generates synthetic test images
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # Project documentation
│
├── images/                          # Input images folder
│   ├── sample_01.jpg
│   ├── sample_02.jpg
│   └── ... (10 images total)
│
├── image_features_dataset.csv       # Output: extracted features + predictions
├── output_plots.png                 # Output: prediction visualization graphs
└── sample_image_comparison.png      # Output: color vs grayscale comparison
```

---

## 5. Methodology

### 5.1 Image Generation

Since real labeled image datasets are not always available, a helper script (`generate_sample_images.py`) was written to synthetically generate 10 test images. Each image is a 200×200 pixel canvas with:

- A random background brightness level (50–220 pixel intensity)
- Random colored rectangles and circles drawn on top

This ensures each image has a unique combination of brightness and edge complexity, making it suitable for regression training.

### 5.2 Image Loading

The `main.py` script supports three input modes:
1. Provide a custom folder path
2. Provide individual image file paths
3. Use the default `images/` folder

Images are loaded using `cv2.imread()` and stored as NumPy arrays.

### 5.3 Feature Extraction

Two features are extracted from each image:

**Feature 1 — Brightness**
- The image is converted from BGR (color) to Grayscale using `cv2.cvtColor()`
- Average pixel intensity is computed using `np.mean(gray)`
- Range: 0 (black) to 255 (white)

**Feature 2 — Edge Count**
- Canny Edge Detection is applied using `cv2.Canny(gray, 100, 200)`
- The number of non-zero (edge) pixels is counted using `np.sum(edges > 0)`
- This reflects the structural complexity of the image

### 5.4 Target Variable Creation

A simulated target variable `price` is computed using the formula:

```
price = (brightness × 0.5) + (edges × 0.01)
```

This formula is deterministic and directly tied to the extracted features, which allows the Linear Regression model to learn the relationship accurately — making it ideal for demonstrating the concept.

### 5.5 Model Training

- Features matrix: `X = [brightness, edges]`
- Target vector: `y = [price]`
- Data split: **80% training / 20% testing** using `train_test_split()`
- Model: `LinearRegression()` from Scikit-learn
- The model learns coefficients (weights) for brightness and edges, and an intercept

The learned model takes the form:

```
predicted_price = (w1 × brightness) + (w2 × edges) + b
```

Where `w1`, `w2` are the learned coefficients and `b` is the intercept.

### 5.6 Model Evaluation

Two standard regression metrics are used:

| Metric | Formula | Meaning |
|---|---|---|
| **MSE** (Mean Squared Error) | avg((actual - predicted)²) | Average squared error; lower is better |
| **R² Score** | 1 - (SS_res / SS_tot) | Proportion of variance explained; 1.0 = perfect |

Since the target is derived directly from the features, the model achieves a very high R² score (close to 1.0), confirming it has correctly learned the relationship.

---

## 6. Code Walkthrough

### main.py — Core Functions

| Function | Description |
|---|---|
| `get_user_image_sources()` | Interactive input: folder, file paths, or default |
| `load_images(folder)` | Loads all supported images from a directory |
| `extract_features(images)` | Computes brightness and edge count per image |
| `create_target(df)` | Adds the simulated `price` column to the DataFrame |
| `train_model(df)` | Trains LinearRegression, prints coefficients and metrics |
| `add_predictions(df, model)` | Runs model on all data, adds `predicted_price` column |
| `plot_results(df, y_test, y_pred)` | Generates and saves two visualization plots |
| `show_sample_image(images)` | Displays one image in color and grayscale side by side |

### generate_sample_images.py

Generates 10 synthetic 200×200 images with random shapes and brightness levels. Uses `np.random.seed(42)` for reproducibility. Saves images to the `images/` folder as `.jpg` files.

---

## 7. Results

### 7.1 Dataset Sample

After running the pipeline, the output CSV (`image_features_dataset.csv`) contains:

| filename | brightness | edges | price | predicted_price |
|---|---|---|---|---|
| sample_01.jpg | 142.35 | 4821 | 119.38 | 119.41 |
| sample_02.jpg | 98.72 | 6103 | 110.39 | 110.35 |
| sample_03.jpg | 175.60 | 3245 | 120.25 | 120.22 |
| ... | ... | ... | ... | ... |

*(Actual values will vary based on the generated images)*

### 7.2 Model Coefficients

The trained model learns coefficients very close to the true formula values:

| Parameter | Expected | Learned (approx.) |
|---|---|---|
| brightness coefficient (w1) | 0.5 | ~0.5 |
| edges coefficient (w2) | 0.01 | ~0.01 |
| intercept (b) | 0.0 | ~0.0 |

### 7.3 Evaluation Metrics

| Metric | Value |
|---|---|
| MSE | ~0.0001 (very low) |
| R² Score | ~1.0 (near perfect) |

### 7.4 Visualizations

Two plots are generated and saved as `output_plots.png`:

**Plot 1 — Actual vs Predicted Price**
A scatter plot where each point represents one image. The red dashed line represents the ideal fit (actual = predicted). Points close to this line indicate accurate predictions.

**Plot 2 — Brightness vs Price**
Shows how brightness correlates with price. Orange dots represent actual prices; green triangles represent predicted prices. The overlap confirms the model has learned the relationship correctly.

---

## 8. How to Run the Project

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate sample images
```bash
python generate_sample_images.py
```

### Step 3 — Run the main pipeline
```bash
python main.py
```

When prompted, enter `3` to use the default `images/` folder.

### Expected Output Files
- `image_features_dataset.csv` — full dataset with predictions
- `output_plots.png` — visualization graphs
- `sample_image_comparison.png` — color vs grayscale comparison

---

## 9. Key Concepts Explained

**Linear Regression**
A supervised machine learning algorithm that models the relationship between input features (X) and a continuous output (y) by fitting the best straight line through the data: `y = wX + b`. It minimizes the sum of squared errors between actual and predicted values.

**Feature Engineering**
The process of transforming raw data (images) into numerical features (brightness, edge count) that a machine learning model can understand and learn from.

**Canny Edge Detection**
An algorithm that detects edges in images by applying Gaussian smoothing to reduce noise, computing intensity gradients, and applying hysteresis thresholding to identify sharp transitions in pixel values.

**R² Score (Coefficient of Determination)**
Measures how well the model explains the variance in the target variable. A score of 1.0 means the model perfectly predicts all values. A score of 0 means the model is no better than predicting the mean.

**Train/Test Split**
Dividing the dataset into a training set (used to fit the model) and a test set (used to evaluate performance on unseen data). This prevents overfitting and gives an honest estimate of model accuracy.

---

## 10. Conclusion

This project successfully demonstrates a complete machine learning pipeline applied to image data:

- Images were loaded and preprocessed using OpenCV
- Two interpretable features (brightness and edge count) were extracted
- A Linear Regression model was trained and achieved near-perfect accuracy (R² ≈ 1.0)
- Results were visualized and exported for analysis

The project highlights the importance of **feature engineering** — the idea that the quality of features fed into a model matters more than the complexity of the model itself. Even a simple algorithm like Linear Regression can achieve excellent results when the features are well-designed and relevant to the target.

**Future improvements** could include:
- Using real-world labeled image datasets (e.g., real estate photos with actual prices)
- Extracting more advanced features (color histograms, texture descriptors, HOG features)
- Experimenting with more complex models (Polynomial Regression, Random Forest, CNN)

---

## 11. References

1. OpenCV Documentation — https://docs.opencv.org
2. Scikit-learn Documentation — https://scikit-learn.org/stable/
3. NumPy Documentation — https://numpy.org/doc/
4. Pandas Documentation — https://pandas.pydata.org/docs/
5. Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence.

---

*Report generated for academic submission — B.Tech Computer Science*
