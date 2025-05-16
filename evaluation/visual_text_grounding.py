from model_wrapper import ModelManager
import json
import logging
import concurrent.futures
import os
import re
from pathlib import Path
from base_task import GenericEvaluationTask
import math
import random

LENGTH = [20,50,100,200,300]
TYPE = ["spatial", "causal"]
class VisualDetailUnderstanding(GenericEvaluationTask):
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
        return dataset
    
    def evaluate_question(self, data, system_prompt):
        result = []
        for len in LENGTH:
            for type in TYPE:
                single_result = self.evaluate_permutation_func(data, system_prompt, self.manager,length=str(len), type=type)
                result.append(single_result)
                        
        return {
            "question_id": data["question_id"],
            "results": result
        }
    
    def evaluate_permutation_func(self, data, system_prompt,manager,length = None, type = None):
        """
        Evaluate a single permutation
        Logic needs to be implemented in subclasses
        """
        # construct the prompt using the permutation

        frame_list = []
        video_id = data["video_id"]
        for image in os.listdir(os.path.join(self.video_path, video_id)):
            if image.endswith(".jpg") or image.endswith(".png"):
                frame_list.append(image)
        # randomly choose adversial_len of the description
        
        if len(frame_list) > 20 :
            print(f"Length of frame {len(frame_list)} > 20, skipping...")
            return {
            "question_id": data["question_id"],
            "original_answer": "",
            "content": "",
            "response_tokens": 0,
            "prompt_tokens": 0
        }

        folder_path = os.path.join(self.video_path, video_id)

        images_path = [os.path.join(folder_path, image_path) for image_path in frame_list if os.path.exists(os.path.join(folder_path, image_path))]
        base64_images = [self.image2base64(image_path) for image_path in images_path]
        # change the order of the options according to the permutation
        description = data[length][type]

        # Generate user prompt
        user_prompt  = f"description: {description}\n"


        content, response_tokens, prompt_tokens = manager.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base64Frames=base64_images
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
            "length": length,
            "type": type,
            # "original_answer": description,
            "content": content,
            "response_tokens": response_tokens,
            "prompt_tokens": prompt_tokens
        }
        return result
    
    def calculate_metrics(self, results):
        # Metrics need to be updated
        total_attempts = 0
        correct_attempts = 0
        for result in results:
            content = result["content"]
            if content == "False":
                correct_attempts += 1
            total_attempts += 1

        accuracy = correct_attempts / total_attempts if total_attempts > 0 else 0
        return total_attempts, correct_attempts, accuracy

