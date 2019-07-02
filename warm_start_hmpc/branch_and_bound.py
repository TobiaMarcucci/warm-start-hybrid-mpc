import numpy as np
from time import time
from pygraphviz import AGraph
from subprocess import call
from os import getcwd

class Node(object):
    '''
    Node of the branch and bound tree.
    '''

    def __init__(self, branch, parent=None, lb=-np.inf, additional=None):
        '''
        A node is uniquely identified by its identifier.
        The identifier must be a dictionary.
        The identifier of the node is given by the union of the identifier of its partent and the branch dictionary.

        Parameters
        ----------
        branch : dict
            Sub-identifier that merged with the identifier of the parent gives the identifier of the child.
        parent : Node or None
            Parent node in the branch and bound tree.
            The parent of the root node is None.
        lb : float or +-np.inf
            Lower bound on the score of the node provided by the user (useful in case of warm start).
        additional : anything
            Any additional info that must be stored in a node.
        '''

        # store data
        self.identifier = branch if parent is None else {**branch, **parent.identifier}
        self.lb = lb if parent is None else max(lb, parent.lb)
        self.additional = additional
        self.integer_feasible = None
        
    def solve(self, solver):
        '''
        Solves the subproblem for this node.

        Parameters
        ----------
        solver : function
            Function that given the identifier of the node solves its subproblem.
            The solver must return the follwoing.
            integer_feasible (bool): True if the subproblem is a feasible solution to the combinatorial problem, False if not.
            lb (float or np.inf): score of the subproblem (np.inf if infeasible).
            additional: any data we want to retrieve after the solution.
        '''

        # solve subproblem (overwrites self.lb and self.additional)
        [self.integer_feasible, self.lb, self.additional] = solver(self.identifier)

class Printer(object):
    '''
    Printer for the branch and bound algorithm.
    '''

    def __init__(self, printing_period):
        '''
        Sores the printing_period and initializes the internal state.

        Parameters
        ----------
        printing_period : float or None
            Maximum amount of time in seconds without printing.
        '''

        # store printing_period and initilize only if not None
        self.printing_period = printing_period
        if printing_period is not None:

            # initialize internal state of the printer
            self.start_time = time()
            self.last_print_time = time()
            self.iteration_count = 0
            self.lb = -np.inf
            self.ub = np.inf

            # others
            self._column_width = 15

    def initialize(self, warm_start, tol):
        '''
        Prints the info on the warm start (if one is given) and the first row of the table.
        Does not print anything if the printing_period is None.

        Parameters
        ----------
        warm_start : list of Nodes
            Root nodes of the branch and bound tree.
        tol : float
            Nonnegative tolerance on the convergence of the branch and bound.
        '''

        # shut down printing if not rquired
        if self.printing_period is not None:
            if warm_start is not None:
                self._warm_start(warm_start)
            self._first_row(tol)

    def _warm_start(self, warm_start):
        '''
        Prints the info regarding the warm start provided to the solver.

        Parameters
        ----------
        warm_start : list of Nodes
            Root nodes of the branch and bound tree.
        '''

        print('Loaded warm start with %d nodes.' % len(warm_start), end='')
        print('Lower bound from warm start is %.3f.' % min([n.lb for n in warm_start]))

    def _first_row(self, tol):
        '''
        Prints the first row of the table, the one with the titles of the columns.

        Parameters
        ----------
        tol : float
            Nonnegative tolerance on the convergence of the branch and bound.
        '''

        print('Optimality tolerance set to %.2e.'%tol)
        print('|', end='')
        print('Updates'.center(self._column_width) + '|', end='')
        print('Time (s)'.center(self._column_width) + '|', end='')
        print('Solved nodes'.center(self._column_width) + '|', end='')
        print('Lower bound'.center(self._column_width) + '|', end='')
        print('Upper bound'.center(self._column_width) + '|')
        print((' ' + '-' * (self._column_width)) * 5)

    def update(self, leaves, ub):
        '''
        Function to be called at each iteration of the branch and bound to update the printer.
        Prints the status of the algorithm only in case one of the follwoing holds.
        The root node is solved.
        A new incumbent has been found after the last call of the function.
        Nothing has been printed in the last printing_period seconds.

        Parameters
        ----------
        leaves : list of nodes
            Current leaves of the branch and bound tree.
        ub : float
            Best upper bound in the branch and bound algorithm at the moment of the call of this method.
        '''

        # shut down printing if not rquired
        if self.printing_period is not None:

            # check if a new row must be printed
            if np.isinf(self.lb):
                updates = 'Root node'
            elif ub < self.ub:
                updates = 'New incumbent'
            elif (time() - self.last_print_time) > self.printing_period:
                updates = ''
            else:
                updates = None

            # update internal state of the printer
            self.iteration_count += 1
            self.lb = min([l.lb for l in leaves])
            self.ub = ub
            if updates is not None:
                self.last_print_time = time()

            # print new row
            self._new_row(updates)

    def _new_row(self, updates):
        '''
        Prints a new row of the table.

        Parameters
        ----------
        updates : string
            Updates to write in the first column of the table.
        '''

        if updates is not None:
            print(' ', end='')
            print(updates.ljust(self._column_width+1), end='')
            print(('%.2f' % (time() - self.start_time)).ljust(self._column_width+1), end='')
            print(('%d' % self.iteration_count).ljust(self._column_width+1), end='')
            print(('%.3e' % self.lb).ljust(self._column_width+1), end='')
            print(('%.3e' % self.ub).ljust(self._column_width+1))

    def finalize(self):
        '''
        Prints the final row in the table and the solution report.
        '''

        # shut down printing if not rquired
        if self.printing_period is not None:

            # print final row in the table
            updates = 'Infeasible' if np.isinf(self.ub) else 'Solution found'
            self._new_row(updates)

            # print nodes and time
            print('\nExplored %d nodes in %.3f seconds:'%(self.iteration_count, time() - self.start_time), end=' ')

            # print optimal value
            if np.isinf(self.ub):
                print('problem is infeasible.')
            else:
                print('optimal solution found with cost %.3e.'%self.ub)
        
class Drawer(object):
    '''
    Drawer of the branch and bound tree.
    '''

    def __init__(self, label):
        '''
        Parameters
        ----------
        label : string
            Name of the graph and the pdf file.
        '''

        # store label and initilize only if not None
        self.label = label
        if label is not None:

            # initialize graph and node appearence
            self.graph = AGraph(label=label, directed=True, strict=True, filled=True)
            self.graph.node_attr['style'] = 'filled'
            self.graph.node_attr['fillcolor'] = 'white'

    def initialize(self, warm_start):
        '''
        Draws the multiple root nodes of the tree in case a warm start is given.

        Parameters
        ----------
        warm_start : list of Nodes
            Root nodes of the branch and bound tree.
        '''

        # continue only if drawing is required and a warm start is provided
        if self.label is not None and warm_start is not None:

            # add one root node per time
            for node in warm_start:
                label = 'Branch: ' + self._indent_identifier(node.identifier)
                label +=  '\nLower bound: %.3f' % node.lb + '\n'
                self.graph.add_node(node.identifier, color='green', label=label)

    def update(self, node, threshold):
        '''
        Adds a node to the tree.

        Parameters
        ----------
        node : instance of Node
            Leaf to be added to the tree.
        threshold : float or np.inf
            Difference of the best upper bound found so far and the solution tolerance.
            What happened in the iteration: 'pruning', 'solution_update', 'branching'.
        '''

        # continue only if drawing is required
        if self.label is not None:

            # node color (based on the pruning criteria)
            if node.lb >= threshold: # pruning
                color = 'red'
            elif node.integer_feasible: # solution update
                color = 'blue'
            else: # branching
                color = 'black'

            # node label
            parent_identifier, branch = self._split_identifier(node.identifier)
            label = 'Branch: ' + self._indent_identifier(branch) + '\n'
            label += 'Lower bound: %.3f' % node.lb + '\n'

            # add node to the graphviz tree
            self.graph.add_node(node.identifier, color=color, label=label)

            # connect node to the parent
            if parent_identifier is not None:
                self.graph.add_edge(parent_identifier, node.identifier)

    def finalize(self, incumbent):
        '''
        Highliths the solution, writes the pdf, and opens the pdf.

        Parameters
        ----------
        incumbent : instance of Node
            Leaf associated with the optimal solution.
        '''

        # continue only if drawing is required
        if self.label is not None:
            if incumbent is not None:
                self._draw_solution(incumbent)
            self._save_and_open()

    def _draw_solution(self, incumbent):
        '''
        Marks the leaf with the optimal solution.

        Parameters
        ----------
        incumbent : instance of Node
            Leaf associated with the optimal solution.
        '''

        # fill incumbent node with green and make the border black again
        self.graph.get_node(incumbent.identifier).attr['color'] = 'black'
        self.graph.get_node(incumbent.identifier).attr['fillcolor'] = 'green'

    def _save_and_open(self):
        '''
        Saves the tree in a pdf file and opens it.
        '''

        # write pdf file
        directory = getcwd() + '/' + self.graph.graph_attr['label'].replace(' ', '_')
        self.graph.write(directory + '.dot')
        self.graph = AGraph(directory + '.dot')
        self.graph.layout(prog='dot')
        self.graph.draw(directory + '.pdf')

        # open pdf file
        call(('open', directory + '.pdf'))

    def _split_identifier(self, identifier):
        '''
        Splits the identifier in the parent identifier and the branch.

        Parameters
        ----------
        identifier : dict
            Dictionary identifying a node.
        '''

        # loop to find parent node in the graphviz tree
        for i in range(1, len(identifier)+1):

            # guess the parent removing last elements from the identifier
            parent_identifier = dict(list(identifier.items())[i:])
            if self.graph.has_node(parent_identifier):

                # if correct get also the branch and return
                branch = dict(list(identifier)[:i])
                return parent_identifier, branch

        # otherwise it is a root node
        return None, identifier

    @staticmethod
    def _indent_identifier(identifier):
        '''
        Breaks an identifier in multiples lines.
        Useful for drawing the warm start.

        Parameters
        ----------
        identifier : dict
            Dictionary identifying a node.
        '''
        return '\n'.join([str(i)[1:-1] for i in identifier.items()])

def branch_and_bound(
        solver,
        candidate_selection,
        brancher,
        tol=0.,
        warm_start=None,
        printing_period=3.,
        draw_label=None,
        **kwargs
        ):
    '''
    Branch and bound solver for combinatorial optimization problems.
    All the printing and drawing functions can be removed from the code if necessary.

    Parameters
    ----------
    solver : function
        Function that given the identifier of the node solves its subproblem.
        (See the docs in the solve method of the Node class.)
    candidate_selection : function
        Function that given a list of nodes and the current incumbent node picks the subproblem (node) to solve next.
    brancher : function
        Function that given the identifier of the (solved) candidate node, returns a branch (dict) for each children.
    tol : float
            Nonnegative tolerance on the convergence of the branch and bound.
    warm_start : list of Nodes
            Root nodes of the branch and bound tree.
    printing_period : float or None
        Period in seconds for printing the status of the solver.
        Set it ot None to shut the log.
    draw_label : string
        Name of the graph and the pdf file.

    Returns
    -------
    additional : unpecified
        Generic container of the data to keep from the solution of the incumbent node.
        It is the solution output provided by solver function when applied to
        the incumbent node.
    leaves : list of nodes
        Leaf nodes of the branch and bound tree at convergence.
    '''

    # initialization
    ub = np.inf
    incumbent = None
    leaves = [Node({})] if warm_start is None else warm_start

    # printing and drawing
    printer = Printer(printing_period)
    drawer = Drawer(draw_label)
    printer.initialize(warm_start, tol)
    drawer.initialize(warm_start)

    while True:

        # termination check
        candidate_nodes = [l for l in leaves if l.lb < ub - tol]
        if not candidate_nodes:
            break

        # selection and solution of candidate node
        working_node = candidate_selection(candidate_nodes)
        working_node.solve(solver)

        # pruning
        if working_node.lb >= ub - tol:
            pass

        # solution update
        elif working_node.integer_feasible:
            incumbent = working_node
            ub = working_node.lb
            
        # branching
        else:
            children = [Node(branch, parent=working_node) for branch in brancher(working_node.identifier)]
            leaves.remove(working_node)
            leaves.extend(children)

        # printing and drawing
        printer.update(leaves, ub)
        drawer.update(working_node, ub-tol)

    # printing and drawing
    printer.finalize()
    drawer.finalize(incumbent)

    return incumbent, leaves

def breadth_first(candidate_nodes):
    '''
    candidate_selection function for the branch and bound algorithm.
    FIFO selection of the nodes.
    Good for proving optimality,bad for finding feasible solutions.

    Parameters
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.

    Returns
    -------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''
    
    return candidate_nodes[0]


def depth_first(candidate_nodes):
    '''
    candidate_selection function for the branch and bound algorithm.
    LIFO selection of the nodes.
    Good for finding feasible solutions, bad for proving optimality.

    Parameters
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.

    Returns
    -------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''

    return candidate_nodes[-1]


def best_first(candidate_nodes):
    '''
    candidate_selection function for the branch and bound algorithm.
    Gets the node whose parent has the lowest cost (in case there are siblings
    picks the first in the list, because np.argmin returns the first).

    Good for proving optimality, bad for finding feasible solutions.

    Parameters
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.

    Returns
    -------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''

    # index of cheapest node
    cheapest = np.argmin([l.lb for l in candidate_nodes])

    return candidate_nodes[cheapest]