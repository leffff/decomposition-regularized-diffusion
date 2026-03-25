import argparse
import os
import pandas as pd
from pytorch_fid import fid_score


def main():
    parser = argparse.ArgumentParser(description='Calculate FID between two folders')
    parser.add_argument('--path1', type=str, help='Path to first folder of images')
    parser.add_argument('--path2', type=str, help='Path to second folder of images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--output_csv', type=str, default='fid_results.csv', help='Path to output CSV file')
    args = parser.parse_args()

    fid_value = fid_score.calculate_fid_given_paths(
        [args.path1, args.path2],
        batch_size=256,
        device=args.device,
        dims=2048
    )
    
    print(f"FID score: {fid_value}")

    new_row = pd.DataFrame([{'path1': args.path1, 'path2': args.path2, 'fid': fid_value}])
    
    if os.path.exists(args.output_csv):
        df = pd.read_csv(args.output_csv)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()