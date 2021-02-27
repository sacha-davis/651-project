import itertools
from pythonds.basic import Stack


class ArithmeticGeneration:
    def __init__(self, max_operators, numbers_included=[[0,1,2,3,4,5,6,7,8,9]], output_file="dataset.tsv"):
        self.operators = ['*', '-', '/', '+']
        self.data = set()
        self.combos_seen = set()
        self.max_operators = max_operators
        self.numbers_included = numbers_included
        self.output_file = output_file


    def infixEquations(self, left, operator, right):
        '''
        Args:   - self
                - left (number): the left operand
                - operator: (str)
                - right (number): the right operand
        Returns: result (str) string form of an infix equation  
        '''

        base = "( {} {} {} )"
        result = base.format(left, operator, right)

        return result


    def infixToPostfix(self,infixexpr):
        '''
        Inputs: infixexpr (str): a string form infix expression.
        This function is taken (almost) verbatim from the following site: https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html
        Returns: (str) string form of converted to post fix expression.
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
            if token not in prec and token != ")":
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


    def infixtoPrefix(self):
        '''
        TODO
        Inputs: infixexpr (str): a string form infix expression.
        Returns: (str) string form of converted to pre fix expression.
        '''
        pass


    def workThruCases(self, left_list, right_list):
        together = [left_list, self.operators, right_list]
        combos = set(itertools.product(*together))
        new_operators = set()
        for c in combos:
            if c not in self.combos_seen:
                self.combos_seen.add(c)
                num_operators = 1
                num_operators += sum([str(c[0]).count(x) for x in self.operators])
                num_operators += sum([str(c[2]).count(x) for x in self.operators])
                if num_operators < self.max_operators:
                    result = self.infixEquations(c[0], c[1], c[2])
                    self.data.add(result)
                    new_operators.add(result)
                elif num_operators == self.max_operators:
                    result = self.infixEquations(c[0], c[1], c[2])
                    self.data.add(result)

        if len(new_operators) > 0:
            self.workThruCases(new_operators, right_list)
            self.workThruCases(left_list, new_operators)
            self.workThruCases(new_operators, new_operators)

        return


    def writeToFile(self, dataset):
        '''
        Function write valid infix, postfix, and evaluations of the expressions to file.
        Input: dataset (set) - set of string infix expressions.
        Returns: None
        '''

        file = open(self.output_file, 'w')
        for result in dataset:
            try:
                evaluation = float(eval(result))
                postfix = self.infixToPostfix(result)
                file.write(result + '\t')
                file.write(postfix + '\t')
                file.write(str(evaluation))
                file.write('\n')
            except:
                pass
        file.close()

        return


    def generate(self):
        '''
        Function works through all possible combinations of given operands and operators.
        Returns: None
        '''

        for operands in self.numbers_included:
            combine = []
            s = [(i, k) for i, k in zip(operands[0::2], operands[1::2])]
            for itself in range(operands[0], operands[-1]+1):
                s.append((itself, itself))
            speed_up = set(itertools.permutations(s, 2))

            c = 0
            for speedy in speed_up:
                self.workThruCases(speedy[0], speedy[1])
                combine.append(self.data)
                self.data = set()
                c += 1

        self.data = set().union(*combine)
        self.writeToFile(self.data)
        print("done")



