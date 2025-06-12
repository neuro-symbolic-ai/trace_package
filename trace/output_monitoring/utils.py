from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from collections import defaultdict, Counter
from .config import OutputMonitoringConfig


def extract_text_from_batch(batch: List[List[torch.Tensor]], tokenizer) -> List[str]:
    """
    Extract text sequences from a batch of tensors.

    Args:
        batch: List of tensors containing input_ids
        tokenizer: Tokenizer for decoding

    Returns:
        List of decoded text strings
    """
    texts = []

    for sequence in batch:
        non_pad_tokens = [token for token in sequence if token != tokenizer.pad_token_id]
        text = tokenizer.decode(non_pad_tokens, skip_special_tokens=True)
        texts.append(text)

    return texts



def compute_tag_accuracy(
        tagged_predictions: List[Tuple[str, str]], # List of tuples (token, tag)
        tagged_true: List[Tuple[str, str]],
        pos_accuracies=None  # could be Counter too
):
    if pos_accuracies is None:
        pos_accuracies = defaultdict(list)
    for (pred_token, pred_tag), (true_token, true_tag) in zip(tagged_predictions, tagged_true):
        pos_accuracies[true_tag].append(pred_tag == true_tag)

    return pos_accuracies




def extract_output_monitoring_data(
        batch: Dict[str, torch.Tensor], # contains true labels
        outputs: torch.Tensor,
        tokenizer,
        config: OutputMonitoringConfig,
        pos_tagger: Optional[callable] = None,
        semantic_tagger: Optional[callable] = None
) -> Dict[str, Dict[str, float]]:
    """
    Extract output monitoring data from model outputs.

    Args:
        model: The transformer model
        batch: Input batch
        outputs: Model outputs (logits)
        tokenizer: Tokenizer for decoding
        config: Configuration object

    Returns:
        Dictionary containing POS and semantic accuracy results
    """
    results = {}
    print(f"Extracting output monitoring data with config: {config}")

    # Get predictions and true tokens
    predicted_tokens = torch.argmax(outputs, dim=-1).cpu().tolist() # predictions = torch.argmax(logits, dim=-1).cpu()
    true_tokens = batch['labels'].cpu().tolist() if 'labels' in batch else batch['input_ids'].cpu().tolist() # just assuming labels are input_ids
    predicted_texts = extract_text_from_batch(predicted_tokens, tokenizer)
    true_texts = extract_text_from_batch(true_tokens, tokenizer)

    if config.track_pos_performance:
        print('tagging predicted texts for POS...')
        pos_accuracies = defaultdict(list)
        # we need to iterate over texts and tokenize them
        tagged_prediction = [pos_tagger.tag_text(text) for text in predicted_texts]
        # tagged_prediction = pos_tagger.tag_text(predicted_texts)
        # tagged_true = pos_tagger.tag_text(true_texts)
        tagged_true = [pos_tagger.tag_text(text) for text in true_texts]
        print('computing POS accuracy...')
        # pos_accuracy = compute_pos_accuracy(predicted_tokens, true_tokens, predicted_texts, config)
        # we need to compute accuracy for each text
        for (pred_text, true_text) in zip(tagged_prediction, tagged_true):
            pos_accuracies = compute_tag_accuracy(pred_text, true_text, pos_accuracies)

        accuracy_dict = {}
        for pos_tag, correct_list in pos_accuracies.items():
            accuracy_dict[pos_tag] = np.mean(correct_list) if correct_list else 0.0
        # pos_accuracy = compute_tag_accuracy(tagged_prediction, tagged_true)
        results['pos_accuracy'] = accuracy_dict

    # Compute semantic accuracy if enabled
    if config.track_semantic_roles:
        semantic_accuracies = defaultdict(list)
        tagged_prediction = [semantic_tagger.tag_text(text) for text in predicted_texts]
        tagged_true = [semantic_tagger.tag_text(text) for text in true_texts]
        for (pred_text, true_text) in zip(tagged_prediction, tagged_true):
            semantic_accuracies = compute_tag_accuracy(pred_text, true_text, semantic_accuracies)

        accuracy_dict = {}
        for pos_tag, correct_list in semantic_accuracies.items():
            accuracy_dict[pos_tag] = np.mean(correct_list) if correct_list else 0.0

        results['semantic_accuracy'] = accuracy_dict

    return results