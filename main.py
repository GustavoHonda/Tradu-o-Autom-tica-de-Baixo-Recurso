from src.pipeline import zero_shot, few_shot, few_shot_reverse, zero_shot_reverse

def main():
    path = "./data/portugues-guarani-tupi antigo.xlsx"
    zero_shot(path = path)
    few_shot_reverse(path = path)
    few_shot(path = path)
    zero_shot_reverse(path = path)
    

if __name__ == "__main__":
    main()