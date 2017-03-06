import argparse
import pandas as pd
from mnist_pytorch import *

def main():
    print('Loading model')
    model = Net()
    if args.cuda:
        model.cuda()
    model.load_state_dict(torch.load(params_filename))

    print('loading data!')
    data_path = '../data/'
    test_set = pickle.load(open(data_path + "test.p", "rb"))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    label_predict = np.array([])
    model.eval()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        temp = output.data.max(1)[1].cpu().numpy().reshape(-1)
        label_predict = np.concatenate((label_predict, temp))

    prediction_df = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    prediction_df.reset_index(inplace=True)
    prediction_df.rename(columns={'index': 'ID'}, inplace=True)

    prediction_df.to_csv('sudaquian_submission.csv', index=False)


if __name__ == '__main__':
    main()
