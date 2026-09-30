import random

def simulate_casino(fair=True, spins=1000):
    results = []

    biased_colour = False 
    if random.random() < 0.5:
        biased_number = random.randint(0, 36)
    else:
        biased_colour = random.choice(["red", "black"])

    for i in range(spins):
        if not fair and biased_colour is False and random.random() < 0.03:
            result = biased_number
        elif not fair and biased_colour and random.random() < 0.03:
            if biased_colour == "red":
                colour = list(range(1, 10, 2)) + list(range(12, 18, 2)) + list(range(19, 28, 2))
            else:
                colour = list(range(2, 10, 2)) + list(range(11, 18, 2)) + list(range(20, 28, 2))
            result = random.choice(colour)
            
        else:
            result = random.randint(0, 36)

        results.append(result)

    return results

def sort_data():
    sessions = []

    for i in range(500):
        results = simulate_casino(fair=True)
        sessions.append(results)

    for j in range(500):
        results = simulate_casino(fair=False)
        sessions.append(results)

    data = []
    red_numbers = list(range(1, 10, 2)) + list(range(12, 18, 2)) + list(range(19, 28, 2))

    for results in sessions:
        counts = [results.count(number) for number in range(37)]
        maximum = max(counts)
        red_count = sum(result in red_numbers for result in results)
        expected_red = len(results) * 18/37
        red_difference = abs(red_count - expected_red)
        data.append([maximum, red_difference])

    return data

