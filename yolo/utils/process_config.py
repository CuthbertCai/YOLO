import configparser

def process_config(conf_file):
    """process configure file to generate common_params, dataset_params, net_params, solver_params

    :param conf_file: configure file path
    :return: common_params, dataset_params, net_params, solver_params
    """
    common_params = {}
    dataset_params = {}
    net_params = {}
    solver_params = {}

    #configure_parser
    config = configparser.ConfigParser()
    config.read(conf_file)

    #sections and options
    for section in config.sections():
        #construct common params
        if section == 'Common':
            for option in config.options(section):
                common_params[option]  = config.get(section, option)
        if section == 'Dataset':
            for option in config.options(section):
                dataset_params[option]  = config.get(section, option)
        if section == 'Net':
            for option in config.options(section):
                net_params[option]  = config.get(section, option)
        if section == 'Solver':
            for option in config.options(section):
                solver_params[option]  = config.get(section, option)

    return common_params, dataset_params, net_params, solver_params
