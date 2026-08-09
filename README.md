# Polynomial Regression - Complete Machine Learning Guide

A complete repository for learning **Polynomial Regression** from the ground up. This project covers theory, mathematics, Python implementation, Scikit-learn workflow, model evaluation, overfitting, regularization, and practical applications.

---

## 📚 About the Project

Polynomial Regression is an extension of Linear Regression that is useful for modeling **non-linear relationships** between variables.

This repository provides a structured learning path covering:

- Mathematical foundations
- Polynomial features
- Model training
- Gradient Descent
- Normal Equation
- Model evaluation
- Overfitting and underfitting
- Bias-variance tradeoff
- Regularization
- Scikit-learn implementation
- Pipeline implementation
- From-scratch implementation
- Real-world applications
- Interview and exam preparation

---

## 🎯 Learning Objectives

By completing this repository, you will understand:

- What Polynomial Regression is
- Why Polynomial Regression is needed
- How Polynomial Regression works mathematically
- How polynomial features are generated
- How polynomial degree affects model complexity
- The difference between Linear and Polynomial Regression
- How to implement Polynomial Regression using Scikit-learn
- How to implement Polynomial Regression from scratch
- How to evaluate regression models
- How overfitting occurs in high-degree polynomial models
- How regularization helps control overfitting
- How to select an appropriate polynomial degree

---

## 📖 Topics Covered

### 1. Introduction

- Introduction to Machine Learning
- Introduction to Regression
- Introduction to Linear Regression
- Limitations of Linear Regression
- Introduction to Polynomial Regression
- Real-world examples
- When to use Polynomial Regression

### 2. What is Polynomial Regression?

- Definition
- Working principle
- Polynomial features
- Relationship between Polynomial Regression and Linear Regression
- Feature engineering concept

### 3. Mathematical Equation

General Polynomial Regression equation:

```text
ŷ = β₀ + β₁X + β₂X² + β₃X³ + ... + βₙXⁿ
```

Where:

- `ŷ` = Predicted value
- `β₀` = Intercept
- `β₁, β₂, ..., βₙ` = Model coefficients
- `X` = Original feature
- `X², X³, ..., Xⁿ` = Polynomial features
- `n` = Polynomial degree

### 4. Polynomial Degree

Different polynomial degrees provide different levels of model complexity.

| Degree | Model | Shape |
|---|---|---|
| 1 | Linear | Straight line |
| 2 | Quadratic | Parabola |
| 3 | Cubic | S-shaped curve |
| 4 | Quartic | More flexible curve |
| 5+ | Higher-order polynomial | Highly flexible |

Choosing a very low degree can cause **underfitting**, while choosing a very high degree can cause **overfitting**.

### 5. Linear vs Polynomial Regression

| Feature | Linear Regression | Polynomial Regression |
|---|---|---|
| Relationship | Linear | Curved / Non-linear |
| Features | Original features | Polynomial features |
| Complexity | Low | Depends on degree |
| Overfitting Risk | Lower | Higher with high degree |
| Interpretability | High | Decreases with degree |
| Curve | Straight line | Flexible curve |

> **Important:** Polynomial Regression is non-linear with respect to `X`, but it remains linear with respect to its coefficients.

---

## 🔧 Polynomial Features

Polynomial features transform the original feature into higher-order features.

For example:

```text
X → X, X², X³
```

Using Scikit-learn:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=3,
    include_bias=False
)

X_poly = poly.fit_transform(X)
```

For degree 3, the features become:

```text
X
X²
X³
```

---

## 🔄 Polynomial Regression Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Polynomial Feature Transformation
   ↓
Linear Regression
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Visualization
   ↓
Model Selection
```

---

## 🧮 Cost Function

Polynomial Regression uses the standard regression loss function.

### Mean Squared Error

```text
MSE = (1/m) Σ(yᵢ - ŷᵢ)²
```

The model attempts to minimize the prediction error between actual and predicted values.

---

## 📐 Least Squares Method

The Ordinary Least Squares method finds the polynomial coefficients that minimize the sum of squared residuals.

```text
SSE = Σ(yᵢ - ŷᵢ)²
```

The best-fitting polynomial is the one that produces the minimum SSE.

---

## 🚀 Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the cost function.

The basic update rule is:

```text
βⱼ := βⱼ - α × ∂J/∂βⱼ
```

Where:

- `βⱼ` = Model parameter
- `α` = Learning rate
- `J` = Cost function

The repository includes a complete implementation of Gradient Descent using NumPy.

---

## ⚠️ Overfitting and Underfitting

### Underfitting

Occurs when the polynomial degree is too low to capture the underlying pattern.

**Characteristics:**

- High training error
- High testing error
- High bias
- Low model complexity

### Good Fit

The model captures the underlying pattern while generalizing well to unseen data.

### Overfitting

Occurs when the polynomial degree becomes too high and the model starts learning noise from the training data.

**Characteristics:**

- Very low training error
- High validation/test error
- High variance
- Excessive model complexity

---

## ⚖️ Bias-Variance Tradeoff

Polynomial degree directly affects the bias-variance tradeoff.

```text
Low Degree
    ↓
High Bias + Low Variance
    ↓
Underfitting

Optimal Degree
    ↓
Balanced Bias + Variance
    ↓
Good Generalization

High Degree
    ↓
Low Bias + High Variance
    ↓
Overfitting
```

---

## 🛡️ Regularization

Regularization helps control overfitting in high-degree Polynomial Regression.

### Ridge Regression

Uses an L2 penalty:

```text
λ Σβⱼ²
```

Example:

```python
from sklearn.linear_model import Ridge
```

### Lasso Regression

Uses an L1 penalty:

```text
λ Σ|βⱼ|
```

Example:

```python
from sklearn.linear_model import Lasso
```

### Elastic Net

Combines both L1 and L2 regularization.

```python
from sklearn.linear_model import ElasticNet
```

---

## 📊 Data Preprocessing

Important preprocessing steps include:

- Handling missing values
- Removing duplicates
- Detecting outliers
- Encoding categorical variables
- Feature scaling
- Train-test splitting
- Polynomial feature transformation

> Always perform the train-test split before fitting transformations to avoid data leakage.

---

## 📏 Feature Scaling

Feature scaling becomes increasingly important as the polynomial degree increases.

### Standardization

```text
X_scaled = (X - mean(X)) / std(X)
```

### Min-Max Normalization

```text
X_scaled = (X - min(X)) / (max(X) - min(X))
```

---

## 📈 Model Evaluation

The repository covers the following regression metrics:

### Mean Absolute Error

```text
MAE = (1/m) Σ|yᵢ - ŷᵢ|
```

### Mean Squared Error

```text
MSE = (1/m) Σ(yᵢ - ŷᵢ)²
```

### Root Mean Squared Error

```text
RMSE = √MSE
```

### R² Score

```text
R² = 1 - (SSres / SStot)
```

### Adjusted R²

Adjusted R² considers the number of predictors in the model and is useful when comparing models with different numbers of polynomial features.

---

## 🧪 Scikit-learn Implementation

Basic implementation:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)

model = LinearRegression()

model.fit(X_poly, y)

y_pred = model.predict(X_poly)
```

---

## 🔗 Pipeline Implementation

Polynomial Regression can also be implemented using a Scikit-learn Pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

model = Pipeline([
    ("poly", PolynomialFeatures(
        degree=8,
        include_bias=False
    )),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=10))
])

model.fit(X_train, y_train)
```

Pipelines help keep preprocessing and model training organized while reducing the risk of inconsistent transformations.

---

## 🧑‍💻 From-Scratch Implementation

The repository also explores Polynomial Regression without relying entirely on Scikit-learn.

Implementation concepts include:

- NumPy
- Polynomial feature generation
- Cost function
- Normal Equation
- Gradient Descent
- Model coefficients
- Predictions
- Model evaluation

This helps understand what happens behind Scikit-learn's high-level APIs.

---

## 🔍 Residual Analysis

Residual analysis is used to understand model errors.

Topics include:

- Residual plots
- Error distribution
- QQ plots
- Detecting systematic patterns
- Checking regression assumptions

A good regression model should generally produce residuals without obvious systematic patterns.

---

## 🔄 Cross-Validation

Cross-validation helps evaluate how well a polynomial model generalizes.

It can be used to compare different polynomial degrees:

```text
Degree 1
Degree 2
Degree 3
Degree 4
...
```

The goal is to choose the degree that provides the best generalization rather than simply the best training performance.

---

## 🎛️ Hyperparameter Tuning

Important hyperparameters include:

- Polynomial degree
- Ridge `alpha`
- Lasso `alpha`
- Elastic Net parameters

Tools such as `GridSearchCV` can be used to find suitable values.

---

## 🌍 Real-World Applications

Polynomial Regression can be applied to problems such as:

- Projectile motion
- Salary growth
- Advertising and sales relationships
- Crop yield prediction
- Fuel efficiency analysis
- House price modeling
- Temperature trends
- Exam score prediction

---

## 💡 Advantages

- Simple to understand
- Easy to implement
- More flexible than Linear Regression
- Can model curved relationships
- Works well with small datasets
- Can be combined with regularization
- Highly interpretable compared with many complex ML models

---

## ⚠️ Disadvantages

- High-degree models can overfit
- Sensitive to outliers
- Polynomial features can increase dimensionality quickly
- Higher degrees can cause numerical instability
- Feature interpretation becomes harder as degree increases
- Not suitable for every type of non-linear relationship

---

## 🧠 Key Takeaways

- Polynomial Regression models curved relationships.
- It creates additional polynomial features from existing features.
- It is still linear with respect to its coefficients.
- Polynomial degree controls model complexity.
- Low degree can lead to underfitting.
- High degree can lead to overfitting.
- Feature scaling becomes important for higher-degree models.
- Regularization can help control overfitting.
- Test and validation performance matter more than training performance.
- Cross-validation is useful for selecting the polynomial degree.

---

## 📚 Learning Resources

- *Introduction to Statistical Learning (ISLR)*
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
- Scikit-learn Documentation
- Andrew Ng Machine Learning Course
- StatQuest

---

## 🚀 Future Improvements

Possible future additions:

- More real-world datasets
- Additional regression projects
- Advanced regularization experiments
- Model comparison
- Automated polynomial degree selection
- More visualization examples
- Deployment examples

---

## 👨‍💻 Author

**Maganpreet Singh**

B.Tech Computer Science & Engineering

Machine Learning | Data Science | Python

---

## ⭐ Support

If you find this repository useful for learning Polynomial Regression, consider giving it a ⭐ on GitHub.

---

**Keep learning. Keep building. Keep experimenting. 🚀**
