import pandas as pd
import numpy as np
import ast 

def export_excel_to_bop_csv(excel_path, output_path):
    df_excel = pd.read_excel(excel_path)
    bop_rows = []
    for _, row in df_excel.iterrows():
        raw_tsfm = row['pred_transformation_obj_in_tgt']
        
        if isinstance(raw_tsfm, str):
            tsfm_matrix = np.array(ast.literal_eval(raw_tsfm))
        else:
            tsfm_matrix = np.array(raw_tsfm)
        if tsfm_matrix.ndim == 3:
            T = tsfm_matrix[0]
        else:
            T = tsfm_matrix
        R = T[:3, :3]
        R_str = " ".join(map(str, R.flatten()))
        t = T[:3, 3] * 1000.0 
        t_str = " ".join(map(str, t.flatten()))

        ob_dir= f'./datasets/BOP/tless/test_primesense/{row["video_id"]:06d}'
        import glob
        color_files = sorted(glob.glob(f'{ob_dir}/rgb/*.png'))
        png_id = color_files[row['frame_id']].split('/')[-1].split('.')[0] 
        bop_rows.append({
            'scene_id': int(row['video_id']),
            'im_id': int(png_id),
            'obj_id': int(row['object_id']),
            'score': row.get('xi', 1.0), 
            'R': R_str,
            't': t_str,
            'time': -1
        })

    df_bop = pd.DataFrame(bop_rows)
    df_bop = df_bop[['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']]
    df_bop.to_csv(output_path, index=False)
    print(f"BOP file save to: {output_path}")

excel_file = './results/pose_estimation_results.xlsx'
bop_csv_file = './results/my-method_tless-test.csv'
export_excel_to_bop_csv(excel_file, bop_csv_file)