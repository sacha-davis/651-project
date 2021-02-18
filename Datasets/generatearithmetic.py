
import sys
import itertools
import re
from pythonds.basic import Stack


operators = ['*', '-', '/', '+']
data = set()
combos_seen = set()
max_operators = 3


def infixEquations(left, operand, right):
    base = "( {} {} {} )"
    result = base.format(left, operand, right)

    return result


def infixToPostfix(infixexpr):
    '''
    This function is taken (almost) verbatim from the following site: https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html

    '''
    prec = {}
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1
    opStack = Stack()
    postfixList = []
    tokenList = infixexpr.split()

    for token in tokenList:
        if token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
            postfixList.append(token)
        elif token == '(':
            opStack.push(token)
        elif token == ')':
            topToken = opStack.pop()
            while topToken != '(':
                postfixList.append(topToken)
                topToken = opStack.pop()
        else:
            while (not opStack.isEmpty()) and \
               (prec[opStack.peek()] >= prec[token]):
                  postfixList.append(opStack.pop())
            opStack.push(token)

    while not opStack.isEmpty():
        postfixList.append(opStack.pop())
    return " ".join(postfixList)


def infixtoPrefix():
    pass


def workThruCases(left_list, right_list):
    global operators, data, max_operators
    print("data",len(data))
    together = [left_list, operators, right_list]
    combos = set(itertools.product(*together))
    new_operators = set()
    for c in combos:
        if c not in combos_seen:
            combos_seen.add(c)
            num_operators = 1
            num_operators += sum([str(c[0]).count(x) for x in operators])
            num_operators += sum([str(c[2]).count(x) for x in operators])
            if num_operators < max_operators:
                result = infixEquations(c[0], c[1], c[2])
                data.add(result)
                new_operators.add(result)
            elif num_operators == max_operators:
                result = infixEquations(c[0], c[1], c[2])
                data.add(result)

    if len(new_operators) > 0:
        workThruCases(new_operators, right_list)
        workThruCases(left_list, new_operators)
        workThruCases(new_operators, new_operators)

    return


def writeToFile(dataset):
    file = open("infix_dataset.tsv", 'w')
    for result in dataset:
        try:
            evaluation = float(eval(result))
            postfix = infixToPostfix(result)
            file.write(result + '\t')
            file.write(postfix + '\t')
            file.write(str(evaluation))
            file.write('\n')
        except:
            pass
    file.close()


def main():
    global data

    operands = list(range(1, 3))
    combine = []
    s = [(i,k) for i,k in zip(operands[0::2], operands[1::2])]
    for itself in range(1,3):
        s.append((itself, itself))
    speed_up = set(itertools.permutations(s,2))
    c = 0
    for speedy in speed_up:
        print("--------------------")
        print("speedy",speedy, "is" ,c, "out of", len(speed_up)-1)
        print("--------------------")
        workThruCases(speedy[0], speedy[1])
        combine.append(data)
        data = set()
        c+=1

        # sys.exit()
    data = set().union(*combine)
    print('results len', len(data))
    writeToFile(data)

main()
