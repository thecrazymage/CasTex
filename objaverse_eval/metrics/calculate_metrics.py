import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inception import InceptionV3
from fid_score import calculate_fid
from kid_score import calculate_kid

DEFAULT_INCEPTION_PATH = "objaverse_eval/assets/pt_inception-2015-12-05-6726825d.pth"
DEFAULT_GT_PATH = "objaverse_eval/renders/ground_truth/frames"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate FID and KID metrics between two directories.')
    parser.add_argument('--method_path',    type=str, required=True,                    help='Path to the method/generated images directory')
    parser.add_argument('--gt_path',        type=str, default=DEFAULT_GT_PATH,          help='Path to the ground truth images directory')
    parser.add_argument('--inception_path', type=str, default=DEFAULT_INCEPTION_PATH,   help='Path to the Inception model weights')

    args = parser.parse_args()
        
    fid = calculate_fid(
        args.gt_path,
        args.method_path,
        inception_path=args.inception_path
    )
    print(f"For method {args.method_path} FID = {fid}")

    kid_mean, kid_std = calculate_kid(
        args.gt_path,
        args.method_path,
        inception_path=args.inception_path
    )
    print(f"For method {args.method_path} KID*1e3 = {kid_mean * 1e3} \\pm {kid_std}")