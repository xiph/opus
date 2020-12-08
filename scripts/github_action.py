class GithubActionsTransformer():
    def __init__(self, name):
        self.actions = {
            'on': [
                "push",
                "pull_request",
                "pull_request_target",
            ],
            "jobs": {
            },
            "name": name
        }
        self.host = {
            'win': 'windows-latest',
            'linux': 'ubuntu-latest',
            'android': 'ubuntu-latest',
            'mac': 'macos-latest',
            'ios': 'macos-latest'}

    def transform(self, configs):
        for config in configs:
            job = self._create_job(config)
            self.actions['jobs'][config['name']] = job
        return self.actions

    def _create_job(self, config):
        job = {}
        job['name'] = config['name']
        job['runs-on'] = self.host[config['host']]
        job['steps'] = [
            {
                "uses": "actions/checkout@v2",
                "with": {
                    "fetch-depth": 0
                }
            }
        ]

        workdir = config['workdir'] if 'workdir' in config else '.'

        if workdir != '.':
            job['steps'].append(
                {
                    "run": "mkdir {}".format(workdir),
                    "name": "Create Work Dir"
                })

        if 'install' in config:
            job['steps'].append(
                {
                    "working-directory": "{}".format(workdir),
                    "run": "{}".format(config['install']),
                    "name": "Install"
                })

        if 'configure' in config:
            job['steps'].append(
                {
                    "working-directory": "{}".format(workdir),
                    "run": "{}".format(config['configure']),
                    "name": "Configure"
                })

        if 'build' in config:
            job['steps'].append(
                {
                    "working-directory": "{}".format(workdir),
                    "run": "{}".format(config['build']),
                    "name": "Build"
                })

        if 'test' in config:
            job['steps'].append(
                {
                    "working-directory": "{}".format(workdir),
                    "run": "{}".format(config['test']),
                    "name": "Test"
                }
            )
        return job
