"""
Sanity check script for BERT masked language modeling assignment.
This script tests basic functionality of the three main functions.
"""

import torch
from transformers import BertTokenizer, BertForPreTraining
from bert_inference import mask_random_tokens, predict_masked_tokens, compute_accuracy

# MODEL_NAME_OR_PATH = "prajjwal1/bert-tiny"
MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"


def test_mask_random_tokens():
    """Test that mask_random_tokens works correctly."""
    print("Testing mask_random_tokens()...")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    sentences = ["The cat sat on the mat."]

    masked_data = mask_random_tokens(sentences, tokenizer, num_masks_per_sentence=1)

    # Check return format
    assert len(masked_data) > 0, "Should return at least one masked sentence"
    assert "original" in masked_data[0], "Should have 'original' key"
    assert "masked_sentence" in masked_data[0], "Should have 'masked_sentence' key"
    assert "ground_truth" in masked_data[0], "Should have 'ground_truth' key"
    assert "[MASK]" in masked_data[0]["masked_sentence"], "Should contain [MASK] token"
    assert len(masked_data[0]["ground_truth"]) == 1, "Should mask exactly 1 token"

    print("✓ mask_random_tokens() passed basic checks\n")


def test_predict_masked_tokens():
    """Test that predict_masked_tokens works correctly."""
    print("Testing predict_masked_tokens()...")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = BertForPreTraining.from_pretrained(MODEL_NAME_OR_PATH)

    # Create a simple masked example
    masked_data = [{
        "original": "The cat sat on the mat.",
        "masked_sentence": "The cat sat on the [MASK]",
        "ground_truth": ["mat"],
        "masked_positions": [5]
    }]

    predictions = predict_masked_tokens(masked_data, model, tokenizer, top_k=5)

    # Check return format
    assert len(predictions) > 0, "Should return predictions"
    assert "top_k_predictions" in predictions[0], "Should have 'top_k_predictions' key"
    assert len(predictions[0]["top_k_predictions"]) == 1, "Should predict 1 token"
    assert isinstance(predictions[0]["top_k_predictions"][0][0], str), "Prediction should be a string"
    assert len(predictions[0]["top_k_predictions"][0]) == 5, "Should have 5 top predictions"

    print("✓ predict_masked_tokens() passed basic checks\n")


def test_end_to_end():
    """Test the complete pipeline."""
    print("Testing end-to-end pipeline...")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = BertForPreTraining.from_pretrained(MODEL_NAME_OR_PATH)

    sentences = ["The cat sat on the mat.", "I love programming."]

    # Step 1: Mask tokens
    masked_data = mask_random_tokens(sentences, tokenizer, num_masks_per_sentence=1)
    assert len(masked_data) > 0, "Should mask at least one sentence"

    # Step 2: Predict
    predictions = predict_masked_tokens(masked_data, model, tokenizer, top_k=5)
    assert len(predictions) == len(masked_data), "Should have same number of predictions as masked data"

    print("✓ End-to-end pipeline passed basic checks\n")


if __name__ == "__main__":
    print("="*60)
    print("Running Sanity Checks for BERT Inference Assignment")
    print("="*60 + "\n")

    try:
        test_mask_random_tokens()
        test_predict_masked_tokens()
        test_end_to_end()

        print("="*60)
        print("All sanity checks passed!")
        print("="*60)
        print("\nNote: These are basic tests and are not exhaustive.")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        print("\nPlease fix the issue and try again.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print("\nPlease check your implementation.")
