from src.pipeline import zero_shot, few_shot

def main():
    path = "./data/portugues-guarani-tupi antigo.xlsx"
    zero_shot(path)
    

if __name__ == "__main__":
    main()