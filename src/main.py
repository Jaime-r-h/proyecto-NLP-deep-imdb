from src.data_processing import load_and_preprocess_data

def main():
    data = load_and_preprocess_data("data/imdb.csv")
    
    print("Número de ejemplos:", len(data))
    print("Tokens:", data[0][0])
    print("Label:", data[0][1])

if __name__ == "__main__":
    main()