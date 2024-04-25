import argparse
import yaml

from utils.templates import setup_dict

parser = argparse.ArgumentParser()
parser.add_argument('pre_setup_yaml', type=str, help="yaml setup file for pre training")
parser.add_argument('adv_setup_yaml', type=str, help="path to derived yaml setup file for adversarial training")


if __name__ == "__main__":
    args = parser.parse_args()


    with open(args.pre_setup_yaml, "r") as f:
        setup = yaml.load(f, Loader=yaml.FullLoader)

    key = setup['model']['name'] + '_adv'

    try:
        adv_setup = setup_dict[key]
    except:
        raise KeyError(f"No setup available for {key}")

    setup['training'] = adv_setup['training']
    setup['discriminator'] = adv_setup['discriminator']
    setup['data']['frames_per_sample'] = 90

    with open(args.adv_setup_yaml, 'w') as f:
        yaml.dump(setup, f)
