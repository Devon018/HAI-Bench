from model_wrapper import ModelManager
import json
import logging
import concurrent.futures
import os
import re
from pathlib import Path
from base_task import GenericEvaluationTask


class TemporalUnderstanding(GenericEvaluationTask):

    def __init__(
        self,
        task_name: str,
        prompt_path: str,
        model: ModelManager,
        data_file: str,
        output_file: str,
        enable_multithread: bool = False  # 新增参数：是否启用多线程
    ):

        super().__init__(task_name, prompt_path, model, data_file, output_file, enable_multithread=enable_multithread)
    def filter_data(self, dataset):

        # 没有过滤规则
        return dataset
    
    def evaluate_question(self, data, system_prompt):

        
        result = self.evaluate_permutation_func(data, system_prompt,self.manager)
        
        return {
            "question_id": data["question_id"],
            "results": result
        }
    
    def evaluate_permutation_func(self, data, system_prompt,manager):
        """
        评估单个排列组合
        需要在子类中实现具体逻辑
        """
        # construct the prompt using the permutation
        video_id = data["video_id"]
        frame_list = data["frame_list"]
        answer = data["answer"]
        folder_path = os.path.join(self.video_path, video_id)

        images_path = [os.path.join(folder_path, image_path) for image_path in frame_list if os.path.exists(os.path.join(folder_path, image_path))]

        base64_images = [self.image2base64(image_path) for image_path in images_path]


        # mapping original options to permuted options
            
        # 生成用户提示
        user_prompt = "Please predict the the emotion and behavior of human and pets in next frame based on the previous frames.\n"

        json_schema = PredictionModel.model_json_schema()
        json_schema["stirct"] = True
        content, response_tokens, prompt_tokens = manager.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base64Frames=base64_images,
            response_format={
                "type":"json_schema",
                "json_schema": json_schema,
            }
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
            "original_answer": answer,
            "content": content,
            "response_tokens": response_tokens,
            "prompt_tokens": prompt_tokens
        }
        return result
    
    def calculate_metrics(self, results):

        total_attempts = 0
        correct_attempts = 0

        for single_question in results:
            single_results = single_question["results"]
            correct = True
            for result in single_results:
                if set(result["content"].strip()) != set(result["original_answer"]):
                    correct = False
                    break
            if correct:
                total_attempts += 1
                correct_attempts += 1
            else:
                total_attempts += 1
        accuracy = correct_attempts / total_attempts if total_attempts > 0 else 0
        return total_attempts, correct_attempts, accuracy

