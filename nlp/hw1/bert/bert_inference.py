"""
BERT Masked Language Modeling Inference

This script demonstrates how to use a pretrained BERT model to predict masked tokens.
"""

import torch
import random
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining
from datasets import load_dataset


def load_short_sentences(tokenizer, n=10):
    dataset = load_dataset("roneneldan/TinyStories", split='validation')
    short_sentences = []
    for data in dataset:
        sentence = data['text']
        n_tokens = len(tokenizer.encode(sentence))
        if n_tokens < 128:
            short_sentences.append(sentence)
        if len(short_sentences) >= n:
            break
    return short_sentences


def mask_random_tokens(sentences, tokenizer, num_masks_per_sentence=1):
    """
    Randomly mask tokens in sentences.

    Args:
        sentences: List of strings
        tokenizer: HuggingFace tokenizer
        num_masks_per_sentence: Number of tokens to mask per sentence

    Returns:
        List of dictionaries with keys:
            - "original": original sentence
            - "masked_sentence": sentence with [MASK] tokens
            - "masked_positions": list of token indices that were masked (num_masks_per_sentence >= 1)
            - "ground_truth": list of actual tokens that were masked (num_masks_per_sentence >= 1)
    """
    assert num_masks_per_sentence >= 1, "At least one mask per sentence is required"
    masked_data = []

    for sentence in sentences:
        encoded = tokenizer(sentence, return_tensors=None, add_special_tokens=False)
        input_ids = encoded['input_ids']
        valid_positions = list(range(0, len(input_ids)))

        ############################################################
        # STUDENT IMPLEMENTATION START
        # 1. randomly sample num_masks_per_sentence token indices to mask
        # 2. replace the tokens at the masked positions with tokenizer.mask_token_id/mask_token
        # 3. decode the tokenized+masked input_ids back to text (keep [MASK] tokens visible)
        #
        # Example input:
        # sentence = "The cat sat on the mat"
        # Example output:
        # masked_positions = [3]
        # masked_sentence = "The cat [MASK] on the mat"
        # ground_truth = ["sat"]
        ############################################################
        # remember num_masks_per_sentence >= 1 (although in this assignment we mostly use 1)
        
        # generate random postions
        masked_positions = []
        for _ in range(num_masks_per_sentence):
            pos = random.choice(valid_positions)
            masked_positions.append(pos)
            valid_positions.remove(pos)

        # replace tokens at masked positions with [MASK]
        masked_input_ids = input_ids.copy()
        ground_truth = []
        for pos in masked_positions:
            ground_truth.append(tokenizer.decode([input_ids[pos]]).strip())
            masked_input_ids[pos] = tokenizer.mask_token_id
        masked_sentence = tokenizer.decode(masked_input_ids)
 
        
        ############################################################
        # STUDENT IMPLEMENTATION END
        ############################################################

        masked_data.append({
            "original": sentence,
            "masked_sentence": masked_sentence,
            "masked_positions": masked_positions,
            "ground_truth": ground_truth
        })
    return masked_data


def predict_masked_tokens(masked_data, model, tokenizer, top_k=5):
    """
    Use BERT to predict masked tokens.

    Args:
        masked_data: Output from mask_random_tokens()
        model: HuggingFace BERT model for masked LM
        tokenizer: HuggingFace tokenizer
        top_k: Number of top predictions to return (default: 5)

    Returns:
        List of dictionaries with keys:
            - "original": original sentence
            - "masked_sentence": sentence with [MASK] tokens
            - "ground_truth": list of actual words
            - "top_k_predictions": list of lists of top-k predicted words
    """
    predictions = []

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        for data in tqdm(masked_data, desc="Predicting masked tokens"):
            masked_sentence = data["masked_sentence"]
            ############################################################
            # STUDENT IMPLEMENTATION START
            # 1. tokenize the masked sentence
            # 2. make a forward pass using BERT (refer to https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForPreTraining to understand the model output)
            # 3. get model's top-k prediction for each masked position
            # 4. decode the predicted token IDs to back to words for better readability
            #
            # Example input:
            # masked_sentence = "The cat [MASK] on the mat"
            # Example output (top-k=5):
            # top_k_predictions = [["sat", "dog", "mouse", "cat", "bird"]]
            ############################################################
            top_k_predictions = []

            encoded = tokenizer(masked_sentence, return_tensors=None, add_special_tokens=False)
            input_ids = encoded['input_ids']

            # make a forward pass using BERT
            input_ids_tensor = torch.tensor([input_ids])  # Add batch dimension
            outputs = model(input_ids_tensor)
            # print("outputs:", outputs)
       

            # get model's top-k prediction for each masked position
            for pos in data["masked_positions"]:
                logits = outputs.prediction_logits[0, pos]  # Get logits for the masked position
                top_k_ids = torch.topk(logits, k=top_k).indices.tolist()  # Get top-k token IDs
                top_k_tokens = [tokenizer.decode([token_id]).strip() for token_id in top_k_ids]  # Decode to tokens
                top_k_predictions.append(top_k_tokens)
            

            ############################################################
            # STUDENT IMPLEMENTATION END
            ############################################################

            predictions.append({
                "original": data["original"],
                "masked_sentence": masked_sentence,
                "ground_truth": data["ground_truth"],
                "top_k_predictions": top_k_predictions
            })
    return predictions


def compute_accuracy(predictions, k=5):
    """
    Compute accuracy and pass@k of predictions.

    Args:
        predictions: Output from predict_masked_tokens()
        k: Number of top predictions to consider for pass@k (default: 5)
    Returns:
        Tuple of (accuracy, pass_at_k) as floats (0-100)
    """
    total = 0
    correct_at_1 = 0
    correct_at_k = 0

    print("\n" + "="*80)
    print("BERT Masked Token Predictions")
    print("="*80 + "\n")

    do_print = len(predictions) < 20
    for i, pred in enumerate(predictions, 1):
        if do_print:
            print(f"[Example {i}]")
            original = pred['original']
            if len(original) > 100:
                original_display = original[:97] + "..."
            else:
                original_display = original
            print(f"Original:  {original_display}")

            masked = pred['masked_sentence']
            if len(masked) > 100:
                masked_display = masked[:97] + "..."
            else:
                masked_display = masked
            print(f"Masked:    {masked_display}")
            print()

        # Compare each prediction to ground truth
        for gt, top_k_preds in zip(
            pred["ground_truth"],
            pred["top_k_predictions"]
        ):
            total += 1
            # Case-insensitive comparison (tokens are already clean)
            gt_clean = gt.strip().lower()
            prediction = top_k_preds[0]  # Top-1 prediction
            pred_clean = prediction.strip().lower()

            # Check if top-1 prediction is correct
            if gt_clean == pred_clean:
                correct_at_1 += 1
                correct_at_k += 1
                match_symbol = "✓"
            else:
                # Check if ground truth is in top-k predictions
                top_k_clean = [p.strip().lower() for p in top_k_preds]
                if gt_clean in top_k_clean:
                    correct_at_k += 1
                    match_symbol = "✗ (in top-5)"
                else:
                    match_symbol = "✗"

            if do_print:
                print(f"  Ground truth:     '{gt}'")
                print(f"  Top-1 prediction: '{prediction}' {match_symbol}")
                top_5_display = ', '.join(f"'{p}'" for p in top_k_preds)
                print(f"  Top-5:            [{top_5_display}]")

        if do_print:
            print("-" * 80)
            print()

    pass_at_1 = (correct_at_1 / total * 100) if total > 0 else 0
    pass_at_k = (correct_at_k / total * 100) if total > 0 else 0

    print("="*80)
    print(f"RESULTS:")
    print(f"  Pass@1 (Accuracy): {correct_at_1}/{total} = {pass_at_1:.2f}%")
    print(f"  Pass@{k}:          {correct_at_k}/{total} = {pass_at_k:.2f}%")
    print("="*80 + "\n")
    return pass_at_1, pass_at_k


def main():
    """
    Main function to demonstrate BERT masked language modeling.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-to-test", type=int, default=10)
    args = parser.parse_args()

    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = BertForPreTraining.from_pretrained("google-bert/bert-base-uncased")
    print("Model loaded successfully!\n")

    # Test sentences.
    sentences = load_short_sentences(tokenizer, n=args.n_to_test)

    random.seed(42)
    print("Masking random tokens...")
    masked_data = mask_random_tokens(sentences, tokenizer, num_masks_per_sentence=1)
    print("Predicting masked tokens...")
    predictions = predict_masked_tokens(masked_data, model, tokenizer, top_k=5)
    print("Computing accuracy...")
    accuracy, pass_at_5 = compute_accuracy(predictions, k=5)
    return


if __name__ == "__main__":
    main()
