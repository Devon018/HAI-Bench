import argparse
import logging
import os
from model_wrapper import ModelManager
from visual_text_grounding import VisualDetailUnderstanding as Task1
from temporal_inference import TemporalUnderstanding as Task2
from causal_chain_completion import CausalChainCompletion as Task3

def setup_logging(model_name):
    """Set up logging"""
    log_dir = f"logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/evaluation.log"),
            logging.StreamHandler()
        ]
    )

def run_evaluation(args):
    """Run all evaluation tasks for the specified model"""
    setup_logging(args.model_name)
    logging.info(f"Starting evaluation for model: {args.model_name}")
    
    
    try:
        # Initialize model
        model = ModelManager(model_name=args.model_name)
        
        # Ensure output directory exists
        os.makedirs(f"outputs/{args.model_name}", exist_ok=True)
                
        
        # Run specific task or all tasks based on task_name argument
        if args.task_name == "task1" or args.task_name == "all":
            # Run Task 1
            prompt_path = f"prompts/visual_text_grounding.txt"
            logging.info(f"Running Task 1 - Model: {args.model_name}")
            task1 = Task1(
                model = model
                task_name="task1",
                prompt_path=prompt_path,
                data_file = "../datasets/visual_text_grounding.json",
                output_file=f"outputs/{args.model_name}/task1.json",
            )
            task1.run()
            
        if args.task_name == "task2" or args.task_name == "all":
            # Run Task 2
            prompt_path = f"prompts/temporal_inference.txt"
            logging.info(f"Running Task 2- Model: {args.model_name}")
            task2 = Task2(
                model = model
                task_name="task2",
                prompt_path=prompt_path,
                data_file = "../datasets/visual_text_grounding.json",
                output_file=f"outputs/{args.model_name}/task2.json",
            )
            task2.run()
            
        if args.task_name == "task3" or args.task_name == "all":
            # Run Task 3
            prompt_path = f"prompts/causal_chain_completion.txt"
            logging.info(f"Running Task 3- Model: {args.model_name}")
            task3 = Task3(
                model = model
                task_name="task3",
                prompt_path=prompt_path,
                data_file = "../datasets/causal_chain_completion.json",
                output_file=f"outputs/{args.model_name}/task3.json",
            )
            task3.run()
        
        # Release model resources after evaluation is complete
        if hasattr(model, 'release_memory'):
            model.release_memory()
        
        logging.info(f"All requested evaluation tasks for model {args.model_name} have been completed")
        return True
        
    except Exception as e:
        logging.error(f"Error evaluating model {args.model_name}: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation tasks")
    parser.add_argument("model_name", help="Name of the model to evaluate")
    parser.add_argument("--task_name", help="Name of the task to run", choices=["task1", "task2", "task3", "all"], default="all")

    args = parser.parse_args()    
    run_evaluation(args)