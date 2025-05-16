from model import ModelManager
import json
import logging
import concurrent.futures
import os
import re
from pathlib import Path
from base_task import GenericEvaluationTask
from typing import List
# Generate and print JSON Schema

class CausalChainCompletion(GenericEvaluationTask):
    """
    Encapsulates the original evaluation process into a reusable task class.
    Can be used in different contexts with different model names, prompt paths, and custom evaluation methods.
    Supports checkpoint resumption.
    """
    def __init__(
        self,
        task_name: str,
        prompt_path: str,
        model: ModelManager,
        data_file: str,
        output_file: str,
        enable_multithread: bool = False  # New parameter: whether to enable multithreading
    ):
        """
        Args:
            task_name: Task name
            prompt_path: System prompt file path
            model_name: Model name
            data_file: Test data file path
            output_file: Result save file path
            enable_multithread: Whether to enable multithreaded processing
        """
        super().__init__(task_name, prompt_path, model, data_file, output_file, enable_multithread=enable_multithread)

    def filter_data(self, dataset):
        """
        Filter the dataset, keeping only samples that need to be evaluated
        """
        # filtered_data = []
        # for data in dataset:
        #     if data["classification"] != "undefined":
        #         filtered_data.append(data)
        # return filtered_data
        return dataset
    
    def evaluate_question(self, data, system_prompt):
        """
        Evaluate a single question
        """
        result = self.evaluate_permutation_func(data, system_prompt, self.manager)
        
        return {
            "question_id": data["question_id"],
            "results": result
        }
    
    def evaluate_permutation_func(self, data, system_prompt, manager):
        """
        Evaluate a single permutation
        Logic needs to be implemented in subclasses
        """
        # construct the prompt using the permutation
        video_id = data["video_id"]
        frame_list = []
        for image in os.listdir(os.path.join(self.video_path, video_id)):
            if image.endswith(".jpg") or image.endswith(".png"):
                frame_list.append(image)

        #  corresponding with limit mm per prompt
        if len(frame_list) > 20 :
            print(f"Length of frame {len(frame_list)} > 20, skipping...")
            return {
            "question_id": data["question_id"],
            "original_answer": "",
            "content": "",
            "response_tokens": 0,
            "prompt_tokens": 0
        }
        nodes_list = data["causal_chain"]["nodes"]
        nodes = []
        for node in nodes_list:
            nodes.append({
                "id": node["id"],
                "type": node["type"],
                "entity": node["entity"],
                "description": node["description"],
            })
        edges_len = len(data["causal_chain"]["edges"])
        edges = data["causal_chain"]["edges"]
        folder_path = os.path.join(self.video_path, video_id)

        images_path = [os.path.join(folder_path, image_path) for image_path in frame_list if os.path.exists(os.path.join(folder_path, image_path))]
        base64_images = [self.image2base64(image_path) for image_path in images_path]

        # Generate user prompt
        user_prompt = f"'available options'：{nodes}\n'edges_len'：{edges_len}\n"
        content, response_tokens, prompt_tokens = manager.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base64Frames=base64_images,

        )
        try:
            if isinstance(content, dict):
                pass
            # Otherwise try to parse it as a JSON string
            else:
                content = json.loads(content)
        except Exception as e:
            print("Error parsing JSON:", content)
            content = content
        result = {
            "question_id": data["question_id"],
            "original_answer": edges,
            "content": content,
            "response_tokens": response_tokens,
            "prompt_tokens": prompt_tokens
        }
        return result
    
    def calculate_metrics(self, results):
        """
        Calculate F1 score based on precision and recall
        """
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        for single_question in results:
            result = single_question["results"]
            
            # Extract ground truth and predictions
            ground_truth = result.get("original_answer", [])
            predictions = []
            
            # Extract predictions from content if available
            try:
                if isinstance(result.get("content"), dict) and "edges" in result["content"]:
                    predictions = result["content"]["edges"]
                elif isinstance(result.get("content"), list):
                    predictions = result["content"]
            except Exception as e:
                print(f"Error extracting predictions: {e}")
                
            # Calculate true positives, false positives, and false negatives
            true_positives = sum(1 for pred in predictions if pred in ground_truth)
            false_positives = sum(1 for pred in predictions if pred not in ground_truth)
            false_negatives = sum(1 for gt in ground_truth if gt not in predictions)
            
            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives
        
        # Calculate precision, recall, and F1
        precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "total_questions": len(results),
            "true_positives": total_true_positives,
            "false_positives": total_false_positives,
            "false_negatives": total_false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

