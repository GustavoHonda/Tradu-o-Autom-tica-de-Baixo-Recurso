import re
import evaluate

def compute_metrics(predictions, references, output_file):
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    references_bleu = [ref[0] if isinstance(ref, list) else ref for ref in references]
    references_bleu = [[ref] for ref in references_bleu]
        

    bleu_result = bleu.compute(
        predictions=predictions,
        references=references_bleu
    )


    chrf1 = chrf.compute(
        predictions=predictions,
        references=references,
        word_order=1
    )
    chrf3 = chrf.compute(
        predictions=predictions,
        references=references,
        word_order=3
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("===== METRICS =====\n")
        f.write(f"BLEU:  {bleu_result['score']}\n")
        f.write(f"chrF1: {chrf1['score']}\n")
        f.write(f"chrF3: {chrf3['score']}\n\n")

        f.write("===== PREDICTIONS =====\n")
        for pred, ref in zip(predictions, references):
            f.write(f"PRED: {pred}\n")
            f.write(f"REF:  {ref}\n")
            f.write("---\n")

    return {
        "bleu": bleu_result,
        "chrf1": chrf1,
        "chrf3": chrf3
    }
