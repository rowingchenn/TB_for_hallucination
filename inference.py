from openai import OpenAI
import os
import json


def main():
    test_dir = "/home/weichenzhang/hallucination/TB_for_hallucination/user_questions/user_questions_2.json"
    with open(test_dir, "r") as f:
        messages = json.load(f)

    for message in messages:
        client = OpenAI(
            # api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
            api_key="",
            base_url="http://10.140.54.13:10099/v1/",
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message,
        )

        print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
