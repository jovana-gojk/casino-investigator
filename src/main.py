from simulate import simulate_casino, sort_data
from model import log_reg_model, des_tree_model, rand_forest_model, iso_forest_model

def main():
    fair_data, cheating_data = sort_data()

    # Supervised learning models to classify fair and cheating sessions
    all_data = fair_data + cheating_data
    print("Data sorted and collected. Starting model training...")
    print("Training Logistic Regression model...")
    log_reg_model(all_data)
    print("Training Decision Tree model...")
    des_tree_model(all_data)
    print("Training Random Forest model...")
    rand_forest_model(all_data)

    # Unsupervised learning model to detect cheating sessions
    print("Training Isolation Forest model...")
    iso_forest_model(fair_data, cheating_data)

if __name__ == "__main__":
    main()