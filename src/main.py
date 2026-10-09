from simulate import simulate_casino, sort_data, extract_features
from model import log_reg_model, des_tree_model, rand_forest_model, iso_forest_model
import os
import time
import random

def game_start():
    print(
    r"""
  ____          _                                        
 / ___|__ _ ___(_)_ __   ___                             
| |   / _` / __| | '_ \ / _ \                            
| |__| (_| \__ \ | | | | (_) |                           
 \____\__,_|___/_|_| |_|\___/_             _             
|_ _|_ ____   _____  ___| |_(_) __ _  __ _| |_ ___  _ __ 
 | || '_ \ \ / / _ \/ __| __| |/ _` |/ _` | __/ _ \| '__|
 | || | | \ V /  __/\__ \ |_| | (_| | (_| | || (_) | |   
|___|_| |_|\_/ \___||___/\__|_|\__, |\__,_|\__\___/|_|   
                               |___/                                                   
    """)
    choice = input("SPIN THE WHEEL? Y/N: ").upper()
    return choice


def wheel_spin():
    print("SPINNING THE WHEEL...")
    frames = [
        r"""
         , - ~ ~ ~ - ,
     , '  \    |  o /  ' ,
   ,       \   |   /       ,
  ,         \  |  /         ,
 ,           \ | /           ,
 ,-------------*-------------,
 ,           / | \           ,
  ,         /  |  \         ,
   ,       /   |   \       ,
     ,    /    |    \   , '
       ' - , _ _ _ ,  '
""",
        r"""
         , - ~ ~ ~ - ,
     , '  \    |    /  ' ,
   ,       \   |   /       ,
  ,         \  |  /         ,
 ,           \ | /        o  ,
 ,-------------*-------------,
 ,           / | \           ,
  ,         /  |  \         ,
   ,       /   |   \       ,
     ,    /    |    \   , '
       ' - , _ _ _ ,  '
""",
        r"""
         , - ~ ~ ~ - ,
     , '  \    |    /  ' ,
   ,       \   |   /       ,
  ,         \  |  /         ,
 ,           \ | /           ,
 ,-------------*-------------,
 ,           / | \           ,
  ,         /  |  \         ,
   ,       /   |   \   o   ,
     ,    /    |    \   , '
       ' - , _ _ _ ,  '
""",
        r"""
         , - ~ ~ ~ - ,
     , '  \    |    /  ' ,
   ,       \   |   /       ,
  ,         \  |  /         ,
 ,           \ | /           ,
 ,-------------*-------------,
 ,           / | \           ,
  ,         /  |  \         ,
   ,       /   |   \       ,
     ,    /  o |    \   , '
       ' - , _ _ _ ,  '
""",
        r"""
         , - ~ ~ ~ - ,
     , '  \    |    /  ' ,
   ,       \   |   /       ,
  ,         \  |  /         ,
 ,           \ | /           ,
 ,-------------*-------------,
 , o         / | \           ,
  ,         /  |  \         ,
   ,       /   |   \       ,
     ,    /    |    \   , '
       ' - , _ _ _ ,  '
""",
        r"""
         , - ~ ~ ~ - ,
     , '  \  o |    /  ' ,
   ,       \   |   /       ,
  ,         \  |  /         ,
 ,           \ | /           ,
 ,-------------*-------------,
 ,           / | \           ,
  ,         /  |  \         ,
   ,       /   |   \       ,
     ,    /    |    \   , '
       ' - , _ _ _ ,  '
""",
    ]
    casino_fair = random.choice([True, False])
    history = simulate_casino(fair=casino_fair, cheating_prob=0.1, spins=1000)

    for i in range(5):
        for frame in frames:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("SPINNING 1000 TIMES...")
            print(frame)
            time.sleep(0.15)

    print("THE WHEEL HAS STOPPED.")
    print("1000 SPINS HAVE BEEN COLLECTED.")
    print("\nMOVE TO CASINO DATA? Y/N:")
    user_input = input().upper()
    if user_input == "Y":
        print("MOVING TO CASINO DATA...")
    else:
        print("CONTINUING TO INVESTIGATION...")
  
    return history, user_input, casino_fair

def color_number(n):
    red = '\033[91m'
    black = '\033[30m'
    green = '\033[92m'
    default = '\033[0m'
    red_numbers = list(range(1, 10, 2)) + list(range(12, 18, 2)) + list(range(19, 28, 2))

    if n == 0:
        return f"{green}{n}{default}"
    if n in red_numbers:
        return f"{red}{n}{default}"
    return f"{black}{n}{default}"

def casino_data(history):
    print("\nCASINO DATA")
    print("-" * 50)
    print("LAST 20 SPINS:")
    print("-" * 50)
    for i in range(0, len(history[-20:]), 10):
        row = history[-20:][i:i+10]
        print("  ".join(color_number(x) for x in row))

    user_input = input("\nSEE DATA FEATURES Y/N: ").upper()

    if user_input == "Y":
        features = extract_features(history)
        print("\nDATA FEATURES")
        print("-" * 50)
        print(f"{'FEATURE':<25} {'VALUE':>10}")
        print("-" * 50)
        print(f"{'MAXIMUM NUMBER FREQ':<25} {features[0]:>10}")
        print(f"{'RED DIFFERENCE':<25} {round(features[1], 1):>10}")
        print(f"{'EVEN DIFFERENCE':<25} {round(features[2], 1):>10}")
        print(f"{'HIGH/LOW DIFFERENCE':<25} {round(features[3], 1):>10}")

    print("\nDO YOU THINK THE CASINO IS FAIR OR CHEATING: ")
    print("1. FAIR")
    print("2. CHEATING")
    your_guess = input("\nYOUR INVESTIGATION: ")
    input("PRESS ENTER TO CONTINUE TO INVESTIGATION...")

    return your_guess

def investigate_casino(history, casino_fair):
    fair_data, cheating_data = sort_data()
    all_data = fair_data + cheating_data
    features = extract_features(history)

    print("\nCHOOSE A MODEL TO INVESTIGATE THE CASINO:")

    print("\nSUPERVISED LEARNING MODELS:")
    print("-" * 50)
    print("1. LOGISTIC REGRESSION")
    print("2. DECISION TREE")
    print("3. RANDOM FOREST")

    print("\nUNSUPERVISED LEARNING MODEL:")
    print("-" * 50)
    print("4. ISOLATION FOREST")
    choice = input("MODEL: \n")

    # Supervised learning models to classify fair and cheating sessions
    if choice == "1":
        model_name = "LOGISTIC REGRESSION"
        model, accuracy = log_reg_model(all_data)
        prediction = model.predict([features])

    elif choice == "2":
        model_name = "DECISION TREE"
        model, accuracy = des_tree_model(all_data)
        prediction = model.predict([features])
        
    elif choice == "3":
        model_name = "RANDOM FOREST"
        model, accuracy = rand_forest_model(all_data)
        prediction = model.predict([features])

    # Unsupervised learning model to detect cheating sessions
    elif choice == "4":
        model_name = "ISOLATION FOREST"
        model, accuracy = iso_forest_model(fair_data, cheating_data)
        prediction = model.predict([features])

    input(f"YOU PICKED {model_name}, ACCURACY: {accuracy:.2%}\nPRESS ENTER TO CONTINUE...")
    if casino_fair:
        casino_truth = "FAIR"
    else:
        casino_truth = "CHEATING"

    print("-" * 50)
    if prediction[0] == 1:
        print("MODEL PREDICTION: CASINO IS CHEATING")

        print(f"CASINO IS... {casino_truth}")
        if casino_fair:
            print("THE MODEL IS WRONG")
        else:
            print("THE MODEL IS RIGHT")
    else:
        print("MODEL PREDICTION: CASINO IS FAIR")
        print(f"CASINO IS... {casino_truth}")
        if casino_fair:
            print("THE MODEL IS RIGHT")
        else:
            print("THE MODEL IS WRONG")

def main():
    choice = game_start()
    your_guess = None
    if choice == "Y":
        history, user_input, casino_fair = wheel_spin()
        if user_input == "Y":
            your_guess = casino_data(history)
        investigate_casino(history, casino_fair)
        if your_guess == "1" and casino_fair or your_guess == "2" and not casino_fair:
            answer = "RIGHT"
        elif your_guess == None:
            answer = "WRONG (NO ANSWER)"
        else:
            answer = "WRONG"
        print(f"YOU WERE... {answer}")
    else:
        print("EXITING THE GAME...")
        return
    
    print("THANKS FOR PLAYING THE CASINO INVESTIGATOR... TILL NEXT TIME...")


if __name__ == "__main__":
    main()