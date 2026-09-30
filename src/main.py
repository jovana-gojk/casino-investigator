from simulate import simulate_casino, sort_data
from model import model

def main():
    data = sort_data()
    model(data)

if __name__ == "__main__":
    main()