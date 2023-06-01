import numpy as np
"""
Code for formatting in LaTeX, mostly tables.
"""

class LaTeXTable():
    
    '''
    table: list of lists for formatted string
    '''
    def __init__(self, table):
        self.table = table
        self.newlines = ['\\\ \n']*(len(table))
        self.vlines = [0]*(len(table[0]) + 1)
        self.caption = None
        self.label = None

    def add_row(self, row, index = -1):
        self.table.insert(index, row)
        self.newlines.insert(index, '\\\ \n')
    
    def add_col(self, col, index = 0):
        for c, r in zip(col, self.table):
            r.insert(index, c)
        self.vlines.insert(index, 0)
    
    '''
    index: index of the row the hline will be under. None for all rows
    '''
    def hline(self, index=0):
        if(index == None):
            self.newlines = ['\\\ \hline \n']*len(self.newlines)
        else:
            self.newlines[index] = '\\\ \hline \n'
    
    '''
    index: index of the column the vline will be to the left of, or rightmost if
    #cols. None for all lines
    '''
    def vline(self, linetype='|', index=None):
        if(index == None):
            if(linetype== '|'):    
                self.vlines = [1]*(len(self.vlines))
            else:
                self.vlines = [0]*(len(self.vlines))
        else:
            if(linetype=='|'):
                self.vlines[index] = 1
            else:
                self.vlines[index] = 0
    
    def make_caption(self):
        return '\caption{' + self.caption + '}'
    def make_label(self):
        return '\label{' + self.label + '}'
    def make_vlines(self):
        rtrn = ''
        for vl in self.vlines:
            if(vl == 0):
                rtrn += 'c'
            else:
                rtrn += '|c'
        return rtrn[:-1]
    
    '''
    valid styles: 'standard', 'plain'
    '''
    def assemble(self, style='standard'):
        rtrn = '\begin{table}\n\t'
        rtrn += self.make_caption() + '\n\t'
        rtrn += self.make_label() + '\n\t'
        rtrn +='\begin{center} \n\t\t'
        rtrn += '\begin{tabular}{%s}\n'%self.make_vlines()
        
        for newline, row in zip(self.newlines, self.table):
            rtrn += '\t\t\t' + ' & '.join(row) + newline + '\n'
        
        rtrn += '\t\t\end{tabular}\n'
        rtrn += '\t\end{center}\n'
        rtrn += '\end{table}'
        return rtrn
    
    def print_table(self, style='standard'):
        print(self.assemble())