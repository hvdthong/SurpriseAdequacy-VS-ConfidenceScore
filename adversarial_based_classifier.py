import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--clf_dsa", "-clf_dsa", help="Classification based on Distance-based Surprise Adequacy", action="store_true"
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.clf_dsa, "Select classification based on metrics"
    print(args)

    if args.clf_dsa:
        print('hello')