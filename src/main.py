from simulate import simulate_casino, sort_data
from model import log_reg_model, des_tree_model, rand_forest_model

def main():
    data = sort_data()
    print("Data sorted and collected. Starting model training...")
    print("Training Logistic Regression model...")
    log_reg_model(data)
    print("Training Decision Tree model...")
    des_tree_model(data)
    print("Training Random Forest model...")
    rand_forest_model(data)

if __name__ == "__main__":
    main()