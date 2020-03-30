import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)
    parser.add_argument(
        "--confidnet", "-confidnet", help="Confidnet score for Test Datasets (IMAGENET)", action="store_true",
    )
    parser.add_argument(
        "--adv_confidnet", "-adv_confidnet", help="Confidnet score for Adversarial Examples", action="store_true"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=16
    )
    parser.add_argument(
        "--model", "-model", help="Model for IMAGENET dataset", type=str, default='vgg16'
    )
    args = parser.parse_args()
    assert args.d in ['imagenet'], "Dataset should be 'imagenet'"
    print(args)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.confidnet
