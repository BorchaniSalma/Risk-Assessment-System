import os
import json

with open('config.json', 'r') as f:
    config = json.load(f)
model_path = os.path.join(config['output_model_path'])


def scores():
    with open(os.path.join(os.getcwd(), model_path, 'latestscore.txt'), 'r') \
            as score_file:
        score = float(score_file.read())
        return ({'score': score})


if __name__ == "__main__":
    print(scores())
