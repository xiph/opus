import argparse
import itertools
import logging
import os
import sys

from ruamel.yaml import YAML  # Github actions needs yaml 1.2

from cmake import CMakeTransformer
from github_action import GithubActionsTransformer

platforms = ['win', 'linux', 'mac', 'ios', 'android']
archs = ['x86', 'x86_64', 'armv7', 'arm64']

build_settings = [
    'custom-modes',
    'fixed-point',
    'float-api',
    'intrinsics',
]

# no reason for disable float api if
# fixed point is disabled, so create special rules for this
# enable fixed-point-debug by default in CI if fixed point is enabled
build_settings_rules = [
    (('fixed-point', False), ('float-api', True)),
]


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--github_action_config_path', default=os.path.abspath(os.path.join(os.path.dirname(
        __file__), '..', '.github', 'workflows')), type=str, help="github action config path (default: %(default)s)")
    parser.add_argument('--log-level', dest='log_level', default='info', help='logging level',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(pathname)s,%(lineno)d] %(message)s',
                        level=getattr(logging, args.log_level.upper()),
                        stream=sys.stdout)

    logging.info('Start')

    configs = generate_configs()

    cmake_transformer = CMakeTransformer()
    cmake_configs = cmake_transformer.transform(configs)

    github_actions_cmake_transformer = GithubActionsTransformer(
        "CMake Build Matrix")
    github_action_cmake_configs = github_actions_cmake_transformer.transform(
        cmake_configs)

    github_action_cmake_yaml_file = os.path.join(
        args.github_action_config_path, 'cmake.yml')
    save_yaml_file(github_action_cmake_configs, github_action_cmake_yaml_file)
    logging.info("Generated github actions file {} with {} configs".format(
        github_action_cmake_yaml_file, len(cmake_configs)))

    custom_configs = []
    custom_configs.append({'name': 'ci_configure_self_test',
                           'host': 'linux',
                           'install': "sudo apt-get install python3 && pip3 install ruamel.yaml",
                           'build': 'python3 ./scripts/ci_generator.py',
                           'test': 'echo if you fail this test it means you did not autogenerate and pushed the generated files && git diff && git diff-index --name-status --exit-code HEAD'
                           })
    custom_configs.append({'name': 'whitespace',
                           'host': 'linux',
                           'test': 'git diff-tree --check origin/master HEAD'
                           })
    custom_configs.append({'name': 'cppcheck',
                           'host': 'linux',
                           'install': 'sudo apt-get install cppcheck',
                           'test': 'cppcheck . --force --std=c99'
                           })

    github_actions_custom_transformer = GithubActionsTransformer(
        "Custom Build Matrix")
    github_action_custom_configs = github_actions_custom_transformer.transform(
        custom_configs)
    github_action_custom_yaml_file = os.path.join(
        args.github_action_config_path, 'custom.yml')
    save_yaml_file(github_action_custom_configs,
                   github_action_custom_yaml_file)

    logging.info("Generated github actions file {} with {} configs".format(
        github_action_custom_yaml_file, len(custom_configs)))

    logging.info('Stop')


def generate_configs():
    logging.info('Generate configs')
    configs = []

    # generate True, False combinations for build settings
    # len(combinations) = 2^len(build_settings)
    combinations = list(itertools.product(
        [False, True], repeat=len(build_settings)))

    # Generate dict based on the combinations
    for idx in range(len(combinations)):
        combinations[idx] = dict(zip(build_settings, combinations[idx]))

    # apply the build settings rules
    for build_settings_rule in build_settings_rules:
        for combination in combinations:
            if combination[build_settings_rule[0][0]] == build_settings_rule[0][1]:
                combination[build_settings_rule[1]
                            [0]] = build_settings_rule[1][1]

    # Remove duplicate combinations after the build settings rules was applied
    combinations = [dict(t) for t in {tuple(d.items()) for d in combinations}]

    # Generate all configt based on the combinations
    for combination in combinations:
        for platform in platforms:
            for arch in archs:
                config = {}
                config['configurations'] = combination
                config['arch'] = arch
                config['platform'] = platform
                configs.append(config)
    return configs


def save_yaml_file(d, yaml_file):
    with open(yaml_file, 'w') as outfile:
        outfile.write(
            "# {} is autogenerated by scripts/ci_generator.py do not edit by hand.\n".format(os.path.basename(yaml_file)))
    yaml = YAML()
    yaml.default_flow_style = False
    with open(yaml_file, 'a') as outfile:
        yaml.dump(d, outfile)


if __name__ == '__main__':
    main()
