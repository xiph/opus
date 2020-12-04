class CMakeTransformer():
    def __init__(self):
        self.settings_dict = {
            'custom-modes': 'OPUS_CUSTOM_MODES',
            'fixed-point': 'OPUS_FIXED_POINT',
            'float-api': 'OPUS_ENABLE_FLOAT_API',
            'intrinsics': 'OPUS_DISABLE_INTRINSICS'
        }
        self.test_settings_dict = {
            'assertions': 'OPUS_ASSERTIONS'
        }
        self.cmake_build = {
            'win': {
                'x86_64': True,
                'x86': True,
                'armv7': False,
                'arm64': False
            },
            'linux': {
                'x86_64': True,
                'armv7': False,
                'arm64': False
            },
            'mac': {
                'x86_64': True
            },
            'android': {
                'x86_64': False,
                'armv7': False,
                'arm64': False
            },
            'ios': {
                'x86_64': False,
                'arm64': False
            }
        }
        self.cmake_test = {
            'win': {
                'x86_64': True,
                'x86': True
            },
            'linux': {
                'x86_64': True
            },
            'mac':  {
                'x86_64': True
            }
        }
        self.cmake_generator = {
            'win': '"Visual Studio 16 2019"',
            'ios': '"Unix Makefiles"'
        }
        self.cmake_platform_build_options = {
            'common': '-DCMAKE_BUILD_TYPE=Release',
            # https://cmake.org/cmake/help/latest/generator/Visual%20Studio%2016%202019.html
            'win': {
                'x86_64': '-A x64',
                'x86': '-A Win32',
                'armv7': '-A ARM',
                'arm64': '-A ARM64'
            },
            'android': {
                # https://developer.android.com/ndk/guides/cmake
                'common': '-DCMAKE_TOOLCHAIN_FILE=${ANDROID_HOME}/ndk-bundle/build/cmake/android.toolchain.cmake',
                'x86_64': '-DANDROID_ABI=x86_64',
                'armv7': '-DANDROID_ABI=armeabi-v7a',
                'arm64': '-DANDROID_ABI=arm64-v8a'
            },
            'ios': {
                # https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#cross-compiling-for-ios-tvos-or-watchos
                'common': '-DCMAKE_SYSTEM_NAME=iOS',
                'x86_64': '-DCMAKE_OSX_ARCHITECTURES=x86_64',
                'arm64': '-DCMAKE_OSX_ARCHITECTURES=arm64'
            }
        }

    def transform(self, configs):
        cmake_configs = []
        for config in configs:
            if self._supported_build_target(config['platform'], config['arch']):
                config_dict = {}
                config_dict['configure'] = self._add_config_step(config)
                config_dict['build'] = 'cmake --build . --config Release'
                config_dict['name'] = self._generate_name(config)
                config_dict['host'] = config['platform']
                if self._supported_test_target(config['platform'], config['arch']):
                    config_dict['test'] = 'ctest -C Release'
                cmake_configs.append(config_dict)

        # let's make sure we generate configs for shared libs as well
        cmake_configs_shared_lib = cmake_configs.copy()
        for config in cmake_configs_shared_lib:
            config['configure'] += ' -DOPUS_BUILD_SHARED_LIBRARY=ON'
            config['name'] += '-shared'
        cmake_configs += cmake_configs_shared_lib

        return sorted(cmake_configs, key=lambda i: i['name'])

    def _generate_name(self, config):
        name = config['platform'] + '-' + config['arch']
        for key, value in config['configurations'].items():
            name = name + '-' + key + '-' + ('on' if value else 'off')
        return name

    def _supported_build_target(self, platform, arch):
        try:
            return self.cmake_build[platform][arch]
        except:
            return False

    def _supported_test_target(self, platform, arch):
        try:
            return self.cmake_test[platform][arch]
        except:
            return False

    def _platform_build_option(self, platform, arch):
        platform_build_options = ''
        try:
            platform_build_options += ' -G ' + self.cmake_generator[platform]
        except:
            pass
        try:
            platform_build_options += " " + \
                self.cmake_platform_build_options['common']
        except:
            pass
        try:
            platform_build_options += " " + \
                self.cmake_platform_build_options[platform]['common']
        except:
            pass
        try:
            platform_build_options += " " + \
                self.cmake_platform_build_options[platform][arch]
        except:
            pass

        return platform_build_options

    def _add_config_step(self, config):
        configure = 'cmake .. -DOPUS_BUILD_PROGRAMS=ON -DOPUS_BUILD_TESTING=ON'

        for key, value in config['configurations'].items():
            configure += ' -D' + self.settings_dict[key] + \
                ('=ON' if value else '=OFF')

        configure += self._platform_build_option(
            config['platform'], config['arch'])

        return configure
