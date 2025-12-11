"""
Regression test for multi-sentence quoted segments.

This test ensures that quoted segments containing multiple sentences
are properly split and all sentences are captured, even when the quote
spans multiple lines.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nltk
from src.pipeline.data_loader import DataLoader

def test_multi_sentence_quote():
    """Test that quotes with multiple sentences are properly split."""
    
    # Create a test file with the problematic quote
    test_content = '''Finally, he thanked us again with: "Thanks guys, that helped us a lot. Your
testimony will go on the record". We looked at each other with shining eyes.'''
    
    # Write to temporary file
    test_file = 'data/test_quote_splitting.txt'
    os.makedirs('data', exist_ok=True)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    try:
        loader = DataLoader(test_file)
        segments = list(loader.load_and_split())
        
        # Find the quoted segment
        quoted_segment = None
        for seg in segments:
            if seg.startswith('"') and 'Thanks guys' in seg:
                quoted_segment = seg
                break
        
        assert quoted_segment is not None, "Quoted segment not found"
        
        # Extract content and tokenize
        if quoted_segment.startswith('"') and quoted_segment.endswith('"'):
            inner_content = quoted_segment[1:-1]
            inner_content = inner_content.replace('\n', ' ').replace('\r', ' ')
            sentences = nltk.sent_tokenize(inner_content)
            
            # Verify both sentences are captured
            assert len(sentences) >= 2, f"Expected at least 2 sentences, got {len(sentences)}"
            
            # Check that both sentences are present
            sentence_texts = [s.strip().lower() for s in sentences]
            has_first = any('thanks guys' in s for s in sentence_texts)
            has_second = any('testimony will go on the record' in s for s in sentence_texts)
            
            assert has_first, "First sentence 'Thanks guys, that helped us a lot.' not found"
            assert has_second, "Second sentence 'Your testimony will go on the record' not found"
            
            print("✓ Test passed: Both sentences in multi-sentence quote are captured")
            return True
        else:
            print("✗ Test failed: Quote format incorrect")
            return False
            
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == '__main__':
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    success = test_multi_sentence_quote()
    sys.exit(0 if success else 1)

