from simulate import simulate_casino, sort_data
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
    print("1. SPIN THE WHEEL")
    print("2. VIEW THE DATA")
    print("3. INVESTIGATE THE CASINO")
    print("4. EXIT")    
    choice = input("CHOOSE AN OPTION: ")
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
    return history


def main():
    choice =game_start()
    if choice == "1":
        wheel_spin()
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