## BERT Masked Language Modeling

In this assignment, you will use a pretrained BERT model to predict masked tokens in sentences and evaluate how well it can guess these masked tokens correctly. **Please avoid modifying any of our stencil code. Doing so may break the autograder!**


### Background

BERT is trained with a masked language modeling objective: given a sentence with some tokens masked out (often replaced with `[MASK]`), predict what the masked tokens should be (see ["10.2.1 Masking Words"](https://web.stanford.edu/~jurafsky/slp3/10.pdf)).

Example:
- Input: "The cat sat on the `[MASK]`"
- BERT's top-3 predictions migth be: "mat", "floor", or "table"

You will implement functions to mask tokens, run BERT inference, and compute accuracy metrics.

### Setup

When you open the `bert` folder, you should see the following files:

- `bert_inference.py` (to fill in) - Main code for tokenization, masking, and inference.
- `sanity_check.py` (provided) - Testing script for sanity checking.

### Logic to Implement

You will implement three functions in `bert_inference.py`. Look for the sections marked with `# STUDENT IMPLEMENTATION START`.

Once you have implemented 1) and 2) below, you can run a basic sanity check with:

```bash
python sanity_check.py
```

#### 1) `mask_random_tokens(sentences, tokenizer, num_masks_per_sentence=1)`

Randomly mask tokens in sentences and return the masked sentences along with ground truth tokens.

**What you need to do:**
- Randomly select token positions to mask
- Replace selected positions with `[MASK]` (using `tokenizer.mask_token_id`/`tokenizer.mask_token`)
- Decode the tokenized+masked input_ids back to text (keep `[MASK]` tokens visible) for evaluation

#### 2) `predict_masked_tokens(masked_data, model, tokenizer, top_k=5)`

Use BERT to predict masked tokens.

**What you need to do:**
- Tokenize each masked sentence
- Get model predictions from `outputs = model(**inputs)` (refer to https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForPreTraining to understand the model output)
- Find what the model is predicting at each masked position
- Extract top-k predicted tokens for each mask position using `torch.topk()`
- Decode token IDs to back to words for better readability


### Running Your Implementation

Test your full implementation with:

```bash
python bert_inference.py
```

If everything is implemented correctly, the program should produce output similar to:

```
Loading BERT model and tokenizer...
Model loaded successfully!

Masking random tokens...
Predicting masked tokens...
Computing accuracy...

================================================================================
BERT Masked Token Predictions
================================================================================

[Example 1]
Original:  Spot. Spot saw the shiny car and said, "Wow, Kitty, your car is so bright and clean!" Kitty smile...
Masked:    spot. spot saw the shiny car and said, " wow, kitty, your car is so bright and clean! " kitty smi...

  Ground truth:     '.'
  Top-1 prediction: '.' ✓
  Top-5:            ['.', ';', '!', '?', '...']
--------------------------------------------------------------------------------

[Example 2]
Original:  Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Her name was Da...
Masked:    once upon a time, in a small yard, there was a small [MASK]. the daisy had a name. her name was d...

  Ground truth:     'daisy'
  Top-1 prediction: 'daisy' ✓
  Top-5:            ['daisy', 'dog', 'girl', 'garden', 'rose']
--------------------------------------------------------------------------------

[Example 3]
Original:  Once upon a time, there was a little girl named Lily. She went on a tour with her family to see a...
Masked:    once upon a [MASK], there was a little girl named lily. she went on a tour with her family to see...

  Ground truth:     'time'
  Top-1 prediction: 'time' ✓
  Top-5:            ['time', 'visit', 'while', 'period', 'land']
--------------------------------------------------------------------------------
...


================================================================================
RESULTS:
  Pass@1 (Accuracy): 6/10 = 60.00%
  Pass@5:          9/10 = 90.00%
================================================================================
```

**Note that the number/masked tokens shown above could fluctuate due to randomness + only testing on 10 sentences.** A better reference for submission would be:

```bash
> python bert_inference.py --n-to-test 256
...some outputs omitted
Computing accuracy...

================================================================================
BERT Masked Token Predictions
================================================================================

================================================================================
RESULTS:
  Pass@1 (Accuracy): 207/256 = 80.86%
  Pass@5:          241/256 = 94.14%
================================================================================
```

---

**In order to receive full credit for this part of the assignment, you will need to**:

- Correctly implement `mask_random_tokens()` and `predict_masked_tokens()`. Make sure they can at least pass the `python sanity_check.py`!
- Include the complete terminal output from running `python bert_inference.py` in your write-up.
- Include the complete terminal output from running `python bert_inference.py --n-to-test 256` in your write-up.
