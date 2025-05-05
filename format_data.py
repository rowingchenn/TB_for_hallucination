def main():
    user_questions_dir = "/home/zhaoyu/Documents/llm_agent_hallucination_data/dataset_all/unexpected_transition/theagentcompany/user_questions"
    for file in os.listdir(user_questions_dir):
        with open(os.path.join(user_questions_dir, file), "r") as f:
            user_questions = json.load(f)

        formatted_user_questions = {
            "task_name": f"user_questions_{file.split('.')[0]}",
            "agent_name": "ReAct-o4-mini",
            "model_config": {
                "model": "o4-mini",
                "max_tokens": 1000,
                "temperature": 0.0,
            },
            "input_step": 0,
            "input": None,
        }
        for question in user_questions:
            pass


if __name__ == "__main__":
    main()
