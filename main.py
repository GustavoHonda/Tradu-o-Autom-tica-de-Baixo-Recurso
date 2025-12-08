from src.pipeline import zero_shot, few_shot

def main():
    path = "./data/portugues-guarani-tupi antigo.xlsx"
    few_shot(path = path)
    

if __name__ == "__main__":
    main()