import random

def simulate_casino(fair=True, cheating_prob=0.03, spins=1000):
    results = []

    if not fair:
        cheating_type = random.choice(["number", "colour", "parity", "high_low"])

        # Set a biased number, colour, parity, or high/low based on the cheating type
        if cheating_type == "number":
            biased_number = random.randint(0, 36)
        elif cheating_type == "colour":
            biased_colour = random.choice(["red", "black"])
        elif cheating_type == "parity":
            biased_parity = random.choice(["even", "odd"])
        elif cheating_type == "high_low":
            biased_high_low = random.choice(["high", "low"])

    for i in range(spins):
        if not fair and random.random() < cheating_prob:

            if cheating_type == "number":
                result = biased_number

            elif cheating_type == "colour":
                if biased_colour == "red":
                    colour = list(range(1, 10, 2)) + list(range(12, 18, 2)) + list(range(19, 28, 2))
                else:
                    colour = list(range(2, 10, 2)) + list(range(11, 18, 2)) + list(range(20, 28, 2))
                result = random.choice(colour)

            elif cheating_type == "parity":
                if biased_parity == "even": 
                    parity = list(range(2, 37, 2))
                else:
                    parity = list(range(1, 36, 2))
                result = random.choice(parity)

            elif cheating_type == "high_low":
                if biased_high_low == "high":
                    high_low = list(range(19, 37))
                else:
                    high_low = list(range(1, 19))
                result = random.choice(high_low)
        else:
            result = random.randint(0, 36)

        results.append(result)

    return results

def sort_data():
    sessions = []

    for i in range(500):
        results = simulate_casino(fair=True, cheating_prob=0.03)
        sessions.append(results)

    for j in range(500):
        results = simulate_casino(fair=False, cheating_prob=0.03)
        sessions.append(results)

    data = []
    red_numbers = list(range(1, 10, 2)) + list(range(12, 18, 2)) + list(range(19, 28, 2))

    for results in sessions:
        # Count the occurrences of each number (0-36) in the results and find the maximum occurrence
        number_freq = [results.count(number) for number in range(37)]
        max_number_freq = max(number_freq)

        # Create a feature of the difference between the number of red results and the expected number of red results
        red_count = sum(result in red_numbers for result in results)
        expected_red = len(results) * 18/37
        red_difference = abs(red_count - expected_red)

        # Create a feature of the difference between the number of even results and the expected number of even results
        even_count = sum(result % 2 == 0 for result in results if result != 0)
        expected_even = len(results) * 18/37
        even_difference = abs(even_count - expected_even)

        # Create a feature of the difference between the number of low results (1-18) and the expected number of low results
        high_count = sum(result in range(19, 37) for result in results)
        expected_high = len(results) * 18/37
        high_difference = abs(high_count - expected_high)

        # Use the features created for different bets on roulette to create a feature vector for each session
        data.append([max_number_freq, red_difference, even_difference, high_difference])

    return data

