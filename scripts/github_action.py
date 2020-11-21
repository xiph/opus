class GithubActionsTransformer():
    def __init__(self, name):
        super(GithubActionsTransformer, self).__init__()
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
        self.workdir = 'build'

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
            }, {
                "run": "mkdir {}".format(self.workdir),
                "name": "Create Work Dir"
            },
            {
                "working-directory": "{}".format(self.workdir),
                "run": "{}".format(config['configure']),
                "name": "Configure"
            },
            {
                "working-directory": "{}".format(self.workdir),
                "run": "{}".format(config['build']),
                "name": "Build"
            },
        ]
        if 'test' in config:
            job['steps'].append(
                {
                    "working-directory": "{}".format(self.workdir),
                    "run": "{}".format(config['test']),
                    "name": "Test"
                }
            )
        return job
