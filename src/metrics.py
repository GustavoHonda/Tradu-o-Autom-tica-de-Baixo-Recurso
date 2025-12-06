import evaluate


def compute_metrics(predictions, references, results_path="translation_results.csv"):
    # BLEU
    bleu = evaluate.load("bleu")

    # ChrF
    chrf = evaluate.load("chrf")

    # predictions = ["Olá, como você está?", "Bom dia"]
    # references = [["Olá, como você está?"], ["Bom dia"]]

    results = bleu.compute(predictions=[p.split() for p in predictions],references=[[r[0].split() for r in ref] for ref in references])
    results_chrf1 = chrf.compute(predictions=predictions, references=[r[0] for r in references], order=1)
    results_chrf3 = chrf.compute(predictions=predictions, references=[r[0] for r in references], order=3)

    print("BLEU score:", results["bleu"])
    print("ChrF1:", results_chrf1["score"])
    print("ChrF3:", results_chrf3["score"])

    df_results = pd.DataFrame({
    "prediction": predictions,
    "reference": references
    })

    df_results["bleu"] = bleu_score
    df_results["chrF1"] = chrf1_score
    df_results["chrF3"] = chrf3_score

    df_results.to_csv(results_path, index=False)
    return results, results_chrf1, results_chrf3