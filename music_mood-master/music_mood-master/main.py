import argparse
import json

from agents import *
from utils.utils import set_class2idx

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/audio_feature_config_csv.json')


def main():
    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    global moods
    moods = config["moods"]

    # set_class2idx(moods)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config["agent"]]
    agent = agent_class(config)
    agent.run()
    agent.finalize()




if __name__ == '__main__':
    main()