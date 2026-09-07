# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------

import time
import torch
import copy

from models.engine import SingleTester
from models.utils.common import get_log_string

from dataset import test_data_loader
from config_test import make_cfg
from model import create_model
from loss import Evaluator

class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)  
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

    def test_step(self, iteration, data_dict):
        # coarse
        output_dict_coarse = self.model(data_dict, {})
        torch.cuda.empty_cache()
        output_dict = {k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) 
               for k,v in output_dict_coarse.items()}
        # fine
        N_fine = output_dict['coarse_est_transform'].shape[0]
        for group_id in range(N_fine):
            output_dict,success_flag = self.model(data_dict, output_dict, group_id)
            torch.cuda.empty_cache()
    
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        message = get_log_string(result_dict=result_dict)
        message += ', Hypothesis_T: {}'.format(len(output_dict['estimated_transform']))
        return message

def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()

if __name__ == '__main__':
    main()
