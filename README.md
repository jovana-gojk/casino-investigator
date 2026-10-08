# Casino Investigator
A machine learning project that investigates whether a casino game is fair or unfair based on statistical outcomes.

The project currently simulates a roulette wheel that can be fair or biased, extracts statistical features from the results, and uses machine learning models to determine whether the casino appears to be fair or cheating

The project also includes a interactive game played within the terminal, where a single casino session is generated, the user can inspect the data and features, make their own prediction, and compare it with the chosen machine learning model used to investigate the casino. 

## Features
- Simulates fair and biased casino game outcomes
- Generates labelled data features for machine learning from statistical data
- Supports multiple type of outcome bias, including single number, colour, parity, and high/low biases
- Trains multiple supervised machine learning classification models (logistic regression, decision tree, random forest)
- Trains an unsupervised machine learning model (isolation forest)
- Evaluates model performance across different cheating methods 
- Provides an interactive terminal-based casino investigation game
- Compares the user's prediction with the model's prediction
- Some fun but relatively simple ASCII art

## Machine Learning Models
The project currently includes:
- Logistic Regression
- Decision Tree
- Random Forest
- Isolation Forest

Each supervised learning model is trained for binary classification of the casino sessions as fair or cheating. The isolation forest is trained on the fair sessions and then used to identify the cheating sessions through anomalies.

## Technologies
- Python
- scikit-learn
- pandas

## Current Process
1. Simulates many roulette sessions with either fair or biased outcomes and defines a single set as a casino session.
2. Extracts statistical features from each casino session.
3. Generate labelled training data containing whether the session is fair or cheating.
4. Trains a selected machine learning model.
5. Generate a new casino session for both player and model investigation.
6. Extract the same features and allow the player to view the data.
7. Use the trained model to predict whether the casino is fair or cheating.
8. Compare the model's and player's prediction with the known outcome from the simulation.

## Running the Project
Clone the repository and install the required packages in requirements.txt via:
```pip install -r requirements.txt```

Run the investigation game via cmd:
```python src/main.py```

## Project Structure
Project is contained with the casino-investigator folder with the src folder, `.gitignore`, `README.md`, and `requirements.txt`.
In the src folder is: `simulate.py`, `model.py`, and `main.py`.

`simulate.py`: Simulates roulette wheel spins, as well as fair and cheating casino sessions. If a casino session is not fair, a random bias is chosen from the pool and that casino session only includes that type of cheating.

`model.py`: Trains the main machine learning models using the simulated fair and cheating data. 

`main.py`: Includes the main game terminal interface and runs each process.

## Limitations
The casino data is simulated, so the models are designed to demonstrate machine learning techniques rather than detect real casino fraud. Currently, cheating probability is implemented via forcing a certain outcome if the random percentage is rolled, instead of raising the probability of the outcome itself.

The simulated cheating mechanisms and statistical features are delibrately simplified to make the problem easier to control and suitable for the scale of the machine learning project.

## Ideas and Further Plans
This project was originally designed and brought to life through the idea of an engaging gameplay loop, where you are tasked as a casino investigator versus the machine learning model, and the company that you work for is trying to replace you. In its current state, the program does play like a game, however plans to extend this idea were thought up through gameplay loops similar to roguelikes and puzzle/detective games. 

Planned next steps involve:
- Improving the accuracy of the cheating probability instead of forcing results
- Increasing the difficulty by using extensively trained models/better models or very low probability of cheating.
- Implemementing a GUI with a dealer and roulette wheel that you can interact with and watch spin.
- Allowing interactive gameplay for the player through items used to better inspect the roulette table for visual signs of cheating (instead of just statistical).
- Adding different casino games other than roulette.
