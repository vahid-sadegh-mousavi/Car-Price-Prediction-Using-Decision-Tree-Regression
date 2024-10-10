### Car Price Prediction Using Decision Tree Regression and Feature Engineering

This project involves predicting car prices based on various features such as the year of manufacture, engine power, fuel type, transmission, and more.

### Key Techniques and Methods Used

1. **Data Cleaning and Preprocessing**:
    - **Missing Value Imputation**: Missing values in important features like mileage, engine size, power, and seating capacity were replaced with the mean of the corresponding column.
    - **Feature Transformation**: 
        - Text-based features, such as mileage (in kmpl) and engine size (in cc), were converted to numerical values by removing units and parsing numbers.
        - Manufacturer names were grouped by region (e.g., Japan, USA, India) to reduce the complexity of the `companyname` feature.
    - **One-Hot Encoding**: Categorical variables such as fuel type, seller type, transmission type, and ownership history were transformed into binary variables using one-hot encoding, making them suitable for machine learning models.

2. **Feature Selection**:
    - Only the most relevant features were selected for the model, including price, engine size, power, year, kilometers driven, fuel type, seller type, transmission, and ownership history.
    - Some redundant or less informative features, such as `torque`, were dropped to improve model performance.

3. **Machine Learning Models**:
    - **Decision Tree Regression**:
        - Several configurations of decision tree regression were applied to model the relationship between car features and prices.
        - **Hyperparameter Tuning**: The model was optimized by adjusting parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` to prevent overfitting and improve generalization.
    - **Linear Regression**: This basic regression model was used to understand the linear relationships between features and car prices, providing a baseline for model performance.
    - **Model Evaluation**:
        - **Mean Squared Error (MSE)**: Used to evaluate the model's accuracy by measuring the average squared difference between actual and predicted prices.
        - **R-squared (R²) Score**: Indicates the proportion of variance in car prices explained by the model.
    
4. **Dimensionality Reduction**:
    - **Correlation Analysis**: A correlation matrix was used to analyze the relationships between variables and reduce multicollinearity. Features with high correlations were either combined or eliminated to improve model efficiency.

5. **Visualization**:
    - Extensive use of **Seaborn** and **Matplotlib** libraries for data visualization:
        - **Joint Plots**: To explore the relationships between continuous variables (e.g., year, engine size, power) and car price.
        - **Count Plots**: For analyzing the distribution of categorical features such as fuel type, transmission type, and ownership status.
    - **Decision Tree Visualization**: The trained decision tree models were visualized to better understand the model's decisions and feature importance.

This combination of regression techniques, feature engineering, and dimensionality reduction provides a comprehensive approach to predicting car prices with both interpretable and high-performing models.
## Dataset
The dataset used is a CSV file (`Car details v3.csv`) containing various details of cars sold in India.

## Features
- `price`: The selling price of the car.
- `year`: The year the car was manufactured.
- `engine`: Engine capacity in cc.
- `power`: Maximum power in bhp.
- `seats`: Number of seats in the car.
- `km_driven`: Total kilometers driven.
- `consumption`: Mileage of the car.
- `companyhome`: Region of the car manufacturer.
- `fuel`: Fuel type (Petrol, Diesel, CNG).
- `seller`: Type of seller (Dealer, Individual).
- `transmission`: Transmission type (Manual).
- `owner`: Number of previous owners.

## Requirements
- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

## Usage
1. Clone the repository
2. Install the required dependencies using `pip install -r requirements.txt`
3. Run the `main.py` to train the model and generate predictions.
4. Visualize the results using Seaborn and Matplotlib.

## License
MIT License
