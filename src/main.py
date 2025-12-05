import nltk
import regex # type: ignore
import sys
from src.models.dummy_model import DummyModel
from src.pipeline.comparator import Comparator
from src.pipeline.logger import DisagreementLogger
from src.pipeline.data_loader import DataLoader

def main():
    # Ensure NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Initialize components
    model_a = DummyModel("Model_A_Benepar", variation=True)
    model_b = DummyModel("Model_B_Adversarial", variation=False)
    comparator = Comparator()
    logger = DisagreementLogger(output_dir="disagreement_logs")

    # Use the file loader
    data_file = "data/ASchoolEssay.txt"
    print(f"Loading data from {data_file}...")
    
    try:
        loader = DataLoader(data_file)
        
        print("Starting adversarial evaluation...")
        
        for segment in loader.load_and_split():
            if not segment:
                continue
                
            # The regex splits into blocks. Some blocks might be multi-sentence text (Group 'other').
            # Others are specific quoted/bracketed blocks.
            # We might want to further split 'other' blocks into sentences, 
            # but keep quoted/bracketed blocks as single "sentences" (or units).
            
            # Simple heuristic: if it looks like a normal text block (not starting with quote/bracket),
            # try to split it into sentences using NLTK.
            # If it starts with quote/bracket, treat as single unit.
            
            is_special_block = segment[0] in ['"', '(', '[', '{']
            
            sub_sentences = []
            if not is_special_block:
                # Split by NLTK
                sub_sentences = nltk.sent_tokenize(segment)
            else:
                sub_sentences = [segment]
                
            for sentence in sub_sentences:
                # Clean up whitespace
                sentence = sentence.replace('\n', ' ').strip()
                if not sentence:
                    continue
                    
                # print(f"Processing: {sentence[:50]}...") # Snippet
                
                res_a = model_a.predict(sentence)
                res_b = model_b.predict(sentence)
                
                if not comparator.compare(res_a, res_b):
                    # print(f" -> Disagreement found!")
                    diff = comparator.find_diff(res_a, res_b)
                    logger.log(sentence, model_a.name, res_a, model_b.name, res_b, diff)
                else:
                    pass
                    # print(f" -> Models agree.")

        logger.save()
        print("Evaluation complete.")
        
    except ImportError:
        print("Error: 'regex' module not found. Please install it via 'pip install regex'.")
    except FileNotFoundError:
        print(f"Error: File {data_file} not found.")

if __name__ == "__main__":
    main()
