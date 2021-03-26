import random

vocab = '0123456789+-/*()'
vocab = list(vocab)

def generateWrong():
    global vocab
    
    equs = set()
    while len(equs) < 30000:
        e = ''
        for s in range(500):
            char = str(random.choice(vocab))
            e += char
            e+= ' '
        try:
            float(eval(e))
        except:
            equs.add(e[:-1])
        

    return equs


def writeToFile(equs):
    with open('incorrect_dataset_len500.tsv', 'w') as f:
        for q in equs:
            s = ''
            s += q
            s += '\n'
            f.write(s)
        f.close()


def main():
    e = generateWrong()
    writeToFile(e)

main()

