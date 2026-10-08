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

    print("SPINNING COMPLETE. RESULTS COLLECTED.")
    print("MOVE TO CASINO DATA? Y/N:")
    user_input = input().upper()
    if user_input == "Y":
        print("MOVING TO CASINO DATA...")
    else:
        print("CONTINUING TO INVESTIGATION...")
  
    return history, user_input, casino_fair

def casino_data(history):
    print("CASINO DATA")
    print("LAST 20 SPINS:")
    for i in range(0, len(history[-20:]), 10):
        print("  ".join(str(x) for x in history[-20:][i:i+10]))

    user_input = input("SEE DATA FEATURES Y/N: ").upper()

    if user_input == "Y":
        features = extract_features(history)
        print("\nDATA FEATURES")
        print("-" * 50)
        print(f"{'FEATURE':<25} {'VALUE':>10}")
        print("-" * 50)
        print(f"{'MAXIMUM NUMBER FREQ':<25} {features[0]:>10}")
        print(f"{'RED DIFFERENCE':<25} {features[1]:>10}")
        print(f"{'EVEN DIFFERENCE':<25} {features[2]:>10}")
        print(f"{'HIGH/LOW DIFFERENCE':<25} {features[3]:>10}")

    input("PRESS ENTER TO CONTINUE TO INVESTIGATION...")

def investigate_casino(history, casino_fair):
    fair_data, cheating_data = sort_data()
    all_data = fair_data + cheating_data
    features = extract_features(history)
    print("CHOOSE A MODEL TO INVESTIGATE THE CASINO:")
    print("SUPERVISED LEARNING MODELS:")
    print("1. LOGISTIC REGRESSION")
    print("2. DECISION TREE")
    print("3. RANDOM FOREST")
    print("UNSUPERVISED LEARNING MODEL:")
    print("4. ISOLATION FOREST")
    choice = input("MODEL: ")

    # Supervised learning models to classify fair and cheating sessions
    if choice == "1":
        model = log_reg_model(all_data)
        prediction = model.predict([features])

    elif choice == "2":
        model = des_tree_model(all_data)
        prediction = model.predict([features])
    elif choice == "3":
        model = rand_forest_model(all_data)
        prediction = model.predict([features])

    # Unsupervised learning model to detect cheating sessions
    elif choice == "4":
        model = iso_forest_model(fair_data, cheating_data)
        prediction = model.predict([features])

    if prediction[0] == 1:
        print("MODEL PREDICTION: CASINO IS CHEATING")
        print(f"CASINO IS... ")
    else:
        print("MODEL PREDICTION: CASINO IS FAIR")

def main():
    choice =game_start()
    if choice == "Y":
        history, user_input, casino_fair = wheel_spin()
        if user_input == "Y":
            casino_data(history)
        investigate_casino(history, casino_fair)
    else:
        print("EXITING THE GAME...")
        return


if __name__ == "__main__":
    main()