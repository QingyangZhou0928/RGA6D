# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------
from typing import Dict
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from models.engine.base_tester import BaseTester
from models.utils.summary_board import SummaryBoard
from models.utils.timer import Timer
from models.utils.common import get_log_string
from models.utils.torch import release_cuda, to_cuda


class SingleTester(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        results = []

        for iteration, data_dict in pbar:
            # on start
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            if data_dict['lengths'][-1][0].item()<=3:
                continue
            self.before_test_step(self.iteration, data_dict)  
            # test step
            torch.cuda.synchronize()
            timer.add_prepare_time()
            start_time = time.time()
            output_dict = self.test_step(self.iteration, data_dict)
            end_time = time.time()
            torch.cuda.synchronize()
            timer.add_process_time()
            # eval step
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            # after step
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict) 
            # logging
            result_dict = release_cuda(result_dict)

            tf0 = data_dict['tf0'].cpu().numpy()
            tf1 = data_dict['tf1'].cpu().numpy()
            video_id = data_dict['video_id']
            mask_id = data_dict['mask_id']
            object_id = data_dict['object_id']
            frame_id = data_dict['frame_id']
            score = data_dict['score'] 

            tsfm = output_dict['estimated_transform']
            tsfm = tsfm[result_dict['top_idx']]
            if hasattr(tsfm, 'tolist'):
                tsfm_list = tsfm.tolist()
            else:
                tsfm_list = tsfm
            pred_transformation_obj_in_tgt = np.linalg.inv(tf1) @ tsfm.cpu().numpy() @ tf0
            if hasattr(pred_transformation_obj_in_tgt, 'tolist'):
                pred_transformation_obj_in_tgt_list = pred_transformation_obj_in_tgt.tolist()
            else:
                pred_transformation_obj_in_tgt_list = pred_transformation_obj_in_tgt

            results.append({
                'object_id': object_id,
                'frame_id': frame_id,
                'video_id': video_id,
                'mask_id': mask_id,
                'pred_transformation_obj-nor_in_tgt-nor': tsfm_list,
                'pred_transformation_obj_in_tgt': pred_transformation_obj_in_tgt_list,
                'RRE': result_dict['RRE'],
                'RTE': result_dict['RTE'],
                'score':score,
                'time':end_time-start_time,
                })

            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            pbar.set_description(message)
            torch.cuda.empty_cache()

        # # save the results
        # df = pd.DataFrame(results)
        # result_path = f'./results/pose_estimation_results.xlsx'
        # df.to_excel(result_path, index=False, engine='openpyxl')

        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)