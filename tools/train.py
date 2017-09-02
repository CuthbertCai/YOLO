import sys
from optparse import OptionParser

sys.path.append('./')

import yolo
parser = OptionParser()
parser.add_option('-c', '--conf', dest='configure', help='configure filename')

(options, args) = parser.parse_args()
conf_file = ''
if options.configure:
    conf_file = str(options.configure)
else:
    print('Please specify --conf configure filename')
    exit(0)

common_params, dataset_params, net_params, solver_params = yolo.utils.process_config.process_config(conf_file)
dataset = eval(dataset_params['name'])(common_params, dataset_params)
net = eval(net_params['name'])(common_params, net_params)
solver = eval(solver_params['name'])(dataset, net, common_params, solver_params)
solver.solve()
