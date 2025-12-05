import nltk
from src.models.dummy_model import DummyModel
from src.pipeline.comparator import Comparator
from src.pipeline.logger import DisagreementLogger

def main():
    # Ensure NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Initialize components
    # Model A: Simulate the "current benepar3 model" with false positives
    model_a = DummyModel("Model_A_Benepar", variation=True)
    
    # Model B: Simulate the "adversarial/correct" model
    model_b = DummyModel("Model_B_Adversarial", variation=False)
    
    comparator = Comparator()
    logger = DisagreementLogger(output_dir="disagreement_logs")

    # Dummy Corpus
    corpus = [
        "mom forgot to buy eggs at lunchtime today because she wanted to bake a cake .",
        "this is a simple sentence .",
        "today is a good day ."
    ]

    print("Starting adversarial evaluation...")
    
    for sentence in corpus:
        print(f"Processing: {sentence}")
        res_a = model_a.predict(sentence)
        res_b = model_b.predict(sentence)
        
        if not comparator.compare(res_a, res_b):
            print(f" -> Disagreement found!")
            diff = comparator.find_diff(res_a, res_b)
            logger.log(sentence, model_a.name, res_a, model_b.name, res_b, diff)
        else:
            print(f" -> Models agree.")

    logger.save()
    print("Evaluation complete.")

if __name__ == "__main__":
    main()

