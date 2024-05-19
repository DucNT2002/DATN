import argparse

from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_type', type=str)
    parser.add_argument('--split_index', type=int, default=0)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--split_file', type=str)
    parser.add_argument('--summary_rate', type=float, default=0.3)
    args = parser.parse_args()

    # Access any argument by its name
    video_type = args.video_type
    split_index = args.split_index
    filename = args.filename
    split_file = args.split_file
    summary_rate = args.summary_rate
    config = get_config(mode='train', video_type=video_type, split_index=split_index, filename=filename, split_file=split_file, summary_rate=summary_rate)
    test_config = get_config(mode='test', video_type=video_type, split_index=split_index, filename=filename, split_file=split_file, summary_rate=summary_rate)

    print(config)
    print(test_config)
    print('split_index:', config.split_index)

    train_loader = get_loader(config.mode, config.split_index, config.filename, config.split_file)
    test_loader = get_loader(test_config.mode, test_config.split_index, config.filename, config.split_file)
    solver = Solver(config, train_loader, test_loader)

    solver.build()
    #solver.evaluate(-1)	# evaluates the summaries generated using the initial random weights of the network 
    solver.train()
