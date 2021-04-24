import matplotlib
import matplotlib.pyplot as plt
import csv

x1 = x2 = x3 = x4 = x5 = [i/54 for i in range(1, 163)]

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

y_bleu_1 = []
y_bleu_2 = []
y_bleu_3 = []
y_bleu_4 = []
y_bleu_5 = []

y_stack_len_1 = []
y_stack_len_2 = []
y_stack_len_3 = []
y_stack_len_4 = []
y_stack_len_5 = []



with open('V_SRNN35_40.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row)
        if 'loss' in row['1']:
            s = row['1']
            l = float(s[s.find('loss: ')+len('loss: '):])
            y_loss_1.append(l)

        if 'ind acc' in row['1']:
            s = row['1']
            a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
            y_acc_1.append(a)
        #
        # if 'avg bleu' in row['1']:
        #     s = row['1']
        #     a = float(s[s.find('avg bleu: ')+len('avg bleu: '):])
        #     y_bleu_1.append(a)
        #
        # if 'stack len avg' in row['1']:
        #     s = row['1']
        #     a = float(s[s.find('stack len avg: ')+len('stack len avg: '):])
        #     y_stack_len_1.append(a)

        if 'loss' in row['2']:
            s = row['2']
            l = float(s[s.find('loss: ')+len('loss: '):])
            y_loss_2.append(l)

        if 'ind acc' in row['2']:
            s = row['2']
            a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
            y_acc_2.append(a)
        #
        # if 'avg bleu' in row['2']:
        #     s = row['2']
        #     a = float(s[s.find('avg bleu: ')+len('avg bleu: '):])
        #     y_bleu_2.append(a)
        #
        if 'stack len avg' in row['2']:
            s = row['2']
            a = float(s[s.find('stack len avg: ')+len('stack len avg: '):])
            y_stack_len_2.append(a)

        #
        # if 'loss' in row['3']:
        #     s = row['3']
        #     l = float(s[s.find('loss: ')+len('loss: '):])
        #     y_loss_3.append(l)
        # #
        # if 'ind acc' in row['3']:
        #     s = row['3']
        #     a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
        #     y_acc_3.append(a)
        #
        # if 'avg bleu' in row['3']:
        #     s = row['3']
        #     a = float(s[s.find('avg bleu: ')+len('avg bleu: '):])
        #     y_bleu_3.append(a)


        # if 'stack len avg' in row['3']:
        #     s = row['3']
        #     a = float(s[s.find('stack len avg: ')+len('stack len avg: '):])
        #     y_stack_len_3.append(a)
        # #
        #
        # if 'loss' in row['4']:
        #     s = row['4']
        #     l = float(s[s.find('loss: ')+len('loss: '):])
        #     y_loss_4.append(l)
        # #
        # if 'ind acc' in row['4']:
        #     s = row['4']
        #     a = float(s[s.find('ind acc avg: ')+len('ind acc avg: '):])
        #     y_acc_4.append(a)
        #
        # if 'avg bleu' in row['4']:
        #     s = row['4']
        #     a = float(s[s.find('avg bleu: ')+len('avg bleu: '):])
        #     y_bleu_4.append(a)

        # if 'stack len avg' in row['4']:
        #     s = row['4']
        #     a = float(s[s.find('stack len avg: ')+len('stack len avg: '):])
        #     y_stack_len_4.append(a)
# x1 = x2 = x3 = x4 = x5 = [i/54 for i in range(len(y_acc_1)+1)]


# plt.plot(x1, y_loss_1, label = "VS2S")
# plt.plot(x2, y_loss_2, label = "RL-SS2S")
# plt.plot(x3, y_loss_3, label = "35-40")
# plt.plot(x3, y_loss_4, label = "35-40")
# plt.plot(x3, y_loss_5, label = "SRNN")

# plt.ylabel('Loss')
# plt.title('Loss For Sequence Length 35-40')
# #
plt.plot(x1,y_acc_1, label = "VS2S")
plt.plot(x2, y_acc_2, label = "RL-SS2S")
# plt.plot(x3, y_acc_3, label = "35-40")
# plt.plot(x3, y_acc_4, label = "35-40")
# # plt.plot(x3, y_acc_4, label = "SRNN_W_Init")
# # plt.plot(x3, y_acc_5, label = "SRNN")
#
plt.ylabel('Individual Accuracy')
plt.title('Individual Accuracy For Sequence Length 35-40')


# plt.plot(x1, y_bleu_1, label = "Sup-SS2S")
# plt.plot(x2, y_bleu_2, label = "SRNN")
# plt.plot(x3, y_bleu_3, label = "SRNN 35-40")
# plt.plot(x3, y_bleu_4, label = "SRNN 35-40")
# # plt.plot(x3, y_bleu_4, label = "SRNN_W_Init")
# # plt.plot(x3, y_bleu_5, label = "SRNN")

# plt.ylabel('BLEU Score')
# plt.title('BLEU Score For Sequence Length 35-40')
#
# plt.plot(x1, y_stack_len_1, label = "35-40")
# plt.plot(x2, y_stack_len_2, label = "35-40")
# plt.plot(x3, y_stack_len_3, label = "35-40")
# plt.plot(x3, y_stack_len_4, label = "35-40")
# # plt.plot(x3, y_stack_len_4, label = "SRNN_W_Init")
# # plt.plot(x3, y_stack_len_5, label = "SRNN")

# plt.ylabel('Stack Length')
# plt.title('V-S2S Stack Length Across All Length')

plt.xlabel('Epochs')

plt.legend()

plt.show()
