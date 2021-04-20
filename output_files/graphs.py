import matplotlib
import matplotlib.pyplot as plt
import csv

x1 = x2 = x3 = x4 = x5 = list(range(1,61))
y_loss_1 = []
y_loss_2 = []
y_loss_3 = []
y_loss_4 = []
y_loss_5 = []

y_acc_1 = []
y_acc_2 = []
y_acc_3 = []
y_acc_4 = []
y_acc_5 = []


with open('output_actionw.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if 'loss' in row['actionweights']:
            s = row['actionweights']
            l = float(s[s.find('loss: ')+len('loss: '):])
            y_loss_1.append(l)

        if 'ind acc' in row['actionweights']:
            s = row['actionweights']
            a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
            y_acc_1.append(a)

        # if 'loss' in row['w xtra loss']:
        #     s = row['w xtra loss']
        #     l = float(s[s.find('loss: ')+len('loss: '):])
        #     y_loss_1.append(l)

        # if 'ind acc' in row['w xtra loss']:
        #     s = row['w xtra loss']
        #     a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
        #     y_acc_1.append(a)

        # if 'loss' in row['wo']:
        #     s = row['wo']
        #     l = float(s[s.find('loss: ')+len('loss: '):])
        #     y_loss_2.append(l)

        # if 'ind acc' in row['wo']:
        #     s = row['wo']
        #     a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
        #     y_acc_2.append(a)

        # if 'loss' in row['SRNN length 15 with intialized weights']:
        #     s = row['SRNN length 15 with intialized weights']
        #     l = float(s[s.find('loss: ')+len('loss: '):])
        #     y_loss_3.append(l)

        # if 'ind acc' in row['SRNN length 15 with intialized weights']:
        #     s = row['SRNN length 15 with intialized weights']
        #     a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
        #     y_acc_3.append(a)

        # if 'loss' in row['Vanilla SRNN length 15 with initialized weights']:
        #     s = row['Vanilla SRNN length 15 with initialized weights']
        #     l = float(s[s.find('loss: ')+len('loss: '):])
        #     y_loss_4.append(l)

        # if 'ind acc' in row['Vanilla SRNN length 15 with initialized weights']:
        #     s = row['Vanilla SRNN length 15 with initialized weights']
        #     a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
        #     y_acc_4.append(a)

        # if 'loss' in row['Vanilla SRNN length 15']:
        #     s = row['Vanilla SRNN length 15']
        #     l = float(s[s.find('loss: ')+len('loss: '):])
        #     y_loss_5.append(l)

        # if 'ind acc' in row['Vanilla SRNN length 15']:
        #     s = row['Vanilla SRNN length 15']
        #     a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
        #     y_acc_5.append(a)

        



plt.plot(x1, y_acc_1, label = "Action Weights")
# plt.plot(x2, y_acc_2, label = "Without")
# plt.plot(x1, y_loss_1, label = "Encoder GRU RNN")
# plt.plot(x2, y_loss_2, label = "SRNN GRU")
# plt.plot(x3, y_loss_3, label = "SRNN_W_Init")
# plt.plot(x3, y_loss_4, label = "VRNN_W_Init")
# plt.plot(x3, y_loss_5, label = "VRNN")

plt.ylabel('Individual Acc')
plt.title('Acc Across Models')

# plt.plot(x1, y_acc_1, label = "Encoder GRU RNN")
# plt.plot(x2, y_acc_2, label = "SRNN GRU")
# # plt.plot(x3, y_acc_3, label = "SRNN_W_Init")
# # plt.plot(x3, y_acc_4, label = "VRNN_W_Init")
# # plt.plot(x3, y_acc_5, label = "VRNN")

# plt.ylabel('Individual Accuracy')
# plt.title('Individual Accuracy Across Models (Length 15)')

plt.xlabel('Iterations')

plt.legend()

plt.show()