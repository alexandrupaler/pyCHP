import sys
import random
import time

import numpy as np

CNOT		= 0
HADAMARD	= 1
PHASE		= 2
MEASURE		= 3

class QProg:
    n = 0               # # of qubits
    T = 0               # # of gates

    a = []         # Instruction opcode
    b = []         # Qubit 1
    c = []         # Qubit 2 (target for CNOT)

    DISPQSTATE = False  # whether to print the state (q for final state only, Q for every iteration)
    DISPTIME = False    # whether to print the execution time
    SILENT = False      # whether NOT to print measurement results
    DISPPROG = False    # whether to print instructions being executed as they're executed
    SUPPRESSM = False   # whether to suppress actual computation of determinate measurement results



class QState:
    # Quantum state
    # To save memory and increase speed, the bits are packed 32 to an long
    n = 0               # # of qubits
    x = None   # (2n+1)*n matrix for stabilizer/destabilizer x bits (there's one "scratch row" at
    z = None   # (2n+1)*n matrix for z bits                                                 the bottom)
    r = None   # Phase bits: 0 for +1, 1 for i, 2 for -1, 3 for -i.  Normally either 0 or 2.
    pw = [int(0)] * 32    # pw[i] = 2^i
    over32 = 0          # floor(n/8)+1


def error(k):
    if k == 0:
        print("Syntax: chp [-options] <filename> [input]\n")
    if k == 1:
        print("File not found\n")
    sys.exit()


def cnot(pointer_q, b, c):
    # Apply a CNOT gate with control b and target c
    q = pointer_q

    b5 = (b >> 5)
    c5 = (c >> 5)

    pwb = q.pw[b & 31]
    pwc = q.pw[c & 31]

    for i in range(2 * q.n):
        if q.x[i][b5] & pwb:
            q.x[i][c5] ^= pwc

        if q.z[i][c5] & pwc:
            q.z[i][b5] ^= pwb

        if (q.x[i][b5] & pwb) and (q.z[i][c5] & pwc) and (q.x[i][c5] & pwc) and (q.z[i][b5] & pwb):
            q.r[i] = (q.r[i]+2) % 4

        if (q.x[i][b5] & pwb) and (q.z[i][c5] & pwc) and not(q.x[i][c5] & pwc) and not(q.z[i][b5] & pwb):
            q.r[i] = (q.r[i]+2) % 4


def hadamard(pointer_q, b):
    # Apply a Hadamard gate to qubit b
    q = pointer_q

    b5 = b >> 5
    pw = q.pw[b & 31]

    for i in range(2 * q.n):
        tmp = q.x[i][b5]
        q.x[i][b5] ^= (q.x[i][b5] ^ q.z[i][b5]) & pw
        q.z[i][b5] ^= (q.z[i][b5] ^ tmp) & pw


        if (q.x[i][b5] & pw == pw) and (q.z[i][b5] & pw == pw):
            q.r[i] = (q.r[i] + 2) % 4


def phase(pointer_q, b):
    # Apply a phase gate (|0> ->|0>, |1> ->i|1>) to qubit b
    q = pointer_q

    b5 = b >> 5
    pw = q.pw[b & 31]

    for i in range(2*q.n):
        if (q.x[i][b5] & pw) and (q.z[i][b5] & pw):
            q.r[i] = (q.r[i] + 2) % 4

        q.z[i][b5] ^= q.x[i][b5] & pw

def rowcopy(pointer_q, i, k):
    # Sets row i equal to row k
    q = pointer_q
    for j in range(q.over32):
         q.x[i][j] = q.x[k][j]
         q.z[i][j] = q.z[k][j]
    q.r[i] = q.r[k]


def rowswap(pointer_q, i, k):
    # Swaps row i and row k
    q = pointer_q

    rowcopy(q, 2 * q.n, k)
    rowcopy(q, k, i)
    rowcopy(q, i, 2 * q.n)

def rowset(pointer_q, i, b):
    # Sets row i equal to the bth observable (X_1,...X_n,Z_1,...,Z_n)\
    q = pointer_q

    for j in range(pointer_q.over32):
        pointer_q.x[i][j] = 0
        pointer_q.z[i][j] = 0
    q.r[i] = 0

    if b < q.n:
        b5 = b >> 5
        b31 = b & 31
        pointer_q.x[i][b5] = pointer_q.pw[b31]
    else:
         b5 = (b - pointer_q.n)>>5
         b31 = (b - pointer_q.n)&31
         pointer_q.z[i][b5] = pointer_q.pw[b31]


# int
def clifford(pointer_q, i, k):
    # Return the phase (0,1,2,3) when row i is LEFT-multiplied by row k
    e = 0 # Power to which i is raised
    q = pointer_q
    for j in range(q.over32):
        for l in range(32):
            pw = q.pw[l]
            if (q.x[k][j] & pw) and (not(q.z[k][j] & pw)): # X
                if (q.x[i][j] & pw) and (q.z[i][j] & pw):
                    e += 1 # XY=iZ
                if (not(q.x[i][j] & pw)) and (q.z[i][j] & pw):
                    e -= 1 # XZ=-iY
            if (q.x[k][j] & pw) and (q.z[k][j] & pw): # Y
                if (not(q.x[i][j] & pw)) and (q.z[i][j] & pw):
                    e += 1 # YZ=iX
                if (q.x[i][j] & pw) and (not(q.z[i][j] & pw)):
                    e -= 1 # YX=-iZ
            if (not(q.x[k][j] & pw)) and (q.z[k][j] & pw): # Z
                 if (q.x[i][j] & pw) and (not(q.z[i][j] & pw)):
                     e += 1 # ZX=iY
                 if ((q.x[i][j] & pw) and (q.z[i][j] & pw)):
                     e -= 1 # ZY=-iX

    e = (e + q.r[i] + q.r[k]) % 4
    if e >= 0:
        return e
    else:
        return e + 4


def rowmult(pointer_q, i, k):
    # Left-multiply row i by row k
    q = pointer_q
    q.r[i] = clifford(q,i,k)
    for j in range(q.over32):
        q.x[i][j] ^= q.x[k][j]
        q.z[i][j] ^= q.z[k][j]


def printstate(pointer_q):
    # Print the destabilizer and stabilizer for state q
    q = pointer_q

    for i in range(2*q.n):
        if i == q.n:
            print("")
            for j in range(q.n + 1):
                print("-", end = '')

        if q.r[i] == 2:
            print("\n-", end = '')
        else:
            print("\n+", end = '')

        for j in range(q.n):
             j5 = j >> 5
             pw = q.pw[j & 31]
             if (not(q.x[i][j5] & pw)) and (not(q.z[i][j5] & pw)):
                 print("I", end = '')
             if (q.x[i][j5] & pw) and (not(q.z[i][j5] & pw)):
                 print("X", end = '')
             if (q.x[i][j5] & pw) and (q.z[i][j5] & pw):
                 print("Y", end = '')
             if (not(q.x[i][j5] & pw)) and (q.z[i][j5] & pw):
                 print("Z", end = '')

    print("")


# int
def measure(pointer_q, b, sup):
    # Measure qubit b
    # Return 0 if outcome would always be 0
    #                 1 if outcome would always be 1
    #                 2 if outcome was random and 0 was chosen
    #                 3 if outcome was random and 1 was chosen
    # sup: True if determinate measurement results should be suppressed, False otherwise

    q = pointer_q

    ran = False
    p = 0 # pivot row in stabilizer
    m = 0 # pivot row in destabilizer

    b5 = b >> 5
    pw = q.pw[b & 31]
    for p in range(q.n): # loop over stabilizer generators
        # if a Zbar does NOT commute with Z_b (the
        # operator being measured), then outcome is random
        if q.x[p + q.n][b5] & pw:
            ran = True
        if ran:
            break

    # If outcome is indeterminate
    if ran:
        rowcopy(q, p, p + q.n) # Set Xbar_p := Zbar_p
        rowset(q, p + q.n, b + q.n) # Set Zbar_p := Z_b
        q.r[p + q.n] = 2*(random.randint(0,1) % 2) # moment of quantum randomness

        for i in range(2*q.n): # Now update the Xbar's and Zbar's that don't commute with
            if (not(i == p)) and (q.x[i][b5] & pw): # Z_b
                rowmult(q, i, p)
        if (q.r[p + q.n]):
            return 3
        else:
            return 2

    # If outcome is determinate
    if not(ran) and not(sup):
        for m in range(q.n): # Before we were checking if stabilizer generators commute
            if (q.x[m][b5]&pw):
                break # with Z_b now we're checking destabilizer generators
        rowcopy(q, 2*q.n, m + q.n)
        for i in range(m+1, q.n):
            if (q.x[i][b5] & pw):
                rowmult(q, 2 * q.n, i + q.n)
        if q.r[2 * q.n]:
            return 1
        else:
            return 0

    return 0



def gaussian(pointer_q):
    # Do Gaussian elimination to put the stabilizer generators in the following form:
    # At the top, a minimal set of generators containing X's and Y's, in "quasi-upper-triangular" form.
    # (Return value = number of such generators = log_2 of number of nonzero basis states)
    # At the bottom, generators containing Z's only in quasi-upper-triangular form.

    q = pointer_q
    i = q.n
    g = 0 # Return value

    for j in range(q.n):
        j5 = j >> 5
        pw = q.pw[j & 31]

        k = i
        for k in range(i, 2 * q.n): # Find a generator containing X in jth column
            if q.x[k][j5] & pw:
                break

        if k < 2*q.n:
             rowswap(q, i, k)
             rowswap(q, i - q.n, k - q.n)
             for k2 in range(i + 1, 2*q.n):
                if q.x[k2][j5] & pw:
                    rowmult(q, k2, i) # Gaussian elimination step
                    rowmult(q, i - q.n, k2 - q.n)
             i += 1

    g = i - q.n

    for j in range(q.n):
        j5 = j >> 5
        pw = q.pw[j & 31]

        k = i
        for k in range(i, 2 * q.n): # Find a generator containing Z in jth column
             if q.z[k][j5] & pw:
                 break

        if k < 2 * q.n:
             rowswap(q, i, k)
             rowswap(q, i - q.n, k - q.n)
             for k2 in range(i + 1, 2*q.n):
                 if q.z[k2][j5] & pw:
                    rowmult(q, k2, i)
                    rowmult(q, i - q.n, k2 - q.n)
             i += 1
    return g


def innerprod(pointer_q1, pointer_q2):
    # Returns -1 if q1 and q2 are orthogonal
    # Otherwise, returns a nonnegative integer s such that the inner product is (1/sqrt(2))^s
    return 0


def printbasisstate(pointer_q):
    # Prints the result of applying the Pauli operator in the "scratch space" of q to |0...0>
    q = pointer_q
    e = q.r[2*q.n]

    for j in range(q.n):
         j5 = j>>5
         pw = q.pw[j & 31]
         if (q.x[2*q.n][j5]&pw) and (q.z[2*q.n][j5]&pw): # Pauli operator is "Y"
                 e = (e+1)%4

    if (e==0):
        print("\n+|", end = '')
    if (e==1):
        print("\n+i|", end = '')
    if (e==2):
        print("\n-|", end = '')
    if (e==3):
        print("\n-i|", end = '')

    for j in range(q.n):
         j5 = j >> 5
         pw = q.pw[j & 31]
         if q.x[2 * q.n][j5] & pw:
             print("1", end = '')
         else:
            print("0", end = '')
    print(">", end = '')


def seed(pointer_q, g):
    # Finds a Pauli operator P such that the basis state P|0...0> occurs with nonzero amplitude in q, and
    # writes P to the scratch space of q.  For this to work, Gaussian elimination must already have been
    # performed on q.  g is the return value from gaussian(q).
    q = pointer_q

    q.r[2*q.n] = 0

    for j in range(q.over32):
         q.x[2*q.n][j] = 0 # Wipe the scratch space clean
         q.z[2*q.n][j] = 0

    for i in range(2*q.n - 1, q.n + g - 1, -1):
         f = q.r[i]
         for j in range(q.n - 1, -1, -1):
            j5 = j >> 5
            pw = q.pw[j & 31]
            if q.z[i][j5] & pw:
                min = j
                if q.x[2*q.n][j5] & pw:
                    f = (f + 2) % 4

         if f == 2:
            j5 = min >> 5
            pw = q.pw[min & 31]
            q.x[2 * q.n][j5] ^= pw # Make the seed consistent with the ith equation


def printket(pointer_q):
    # Print the state in ket notation (warning: could be hugenot)
    # log_2 of number of nonzero basis states
    q = pointer_q

    g = gaussian(q)
    print("2^%ld nonzero basis states" % g)
    if g > 31:
         print("State is WAY too big to print")
         return

    seed(q, g)
    printbasisstate(q)
    for t in range(q.pw[g] - 1):
        t2 = t ^ (t + 1) # TODO: Repair XOR
        for i in range(g):
            if t2 & q.pw[i]:
                rowmult(q, 2*q.n, q.n + i)
        printbasisstate(q)
    print("\n")

def difftime(currenttime, starttime):
    # these should be floats
    return currenttime - starttime

def runprog(pointer_h, pointer_q):
    # Simulate the quantum circuit

    h = pointer_h
    q = pointer_q

    # read time
    tp = time.time()
    dt = float(0)
    mvirgin = True

    for t in range(h.T):
        if h.a[t] == CNOT:
            cnot(q, h.b[t], h.c[t])

        if h.a[t] == HADAMARD:
            hadamard(q, h.b[t])

        if h.a[t] == PHASE:
            phase(q, h.b[t])

        if h.a[t] == MEASURE:
            if mvirgin and h.DISPTIME:
                dt = difftime(time.time(), tp)
                print("Gate time: %lf seconds" % dt)
                print("Time per 10000 gates: %lf seconds" % (dt * float(10000)/(h.T - h.n)))
                tp = time.time()
            mvirgin = False

            m = measure(q,h.b[t], h.SUPPRESSM)
            if not h.SILENT:
                print("Outcome of measuring qubit %ld: " % h.b[t])
                if m > 1:
                    print("%d (random)" % (m-2))
                else:
                    print("%d" % m)

            if h.DISPPROG:
                if h.a[t] == CNOT:
                    print("CNOT %ld.%ld" % (h.b[t], h.c[t]))
                if h.a[t] == HADAMARD:
                    print("Hadamard %ld" % h.b[t])
                if h.a[t] == PHASE:
                    print("Phase %ld" % h.b[t])
    print("\n")

    if h.DISPTIME:
         dt = difftime(time.time(), tp)
         print("Measurement time: %lf seconds", dt)
         print("Time per 10000 measurements: %lf seconds\n" % (dt * float(10000)/h.n))

    if h.DISPQSTATE:
         print("Final state:")
         printstate(q)
         gaussian(q)
         printstate(q)
         printket(q)
    return



def preparestate(pointer_q, string_s):
    # Prepare the initial state's "input"
    s = string_s
    q = pointer_q

    l = len(s)
    for b in range(l):
        if s[b] == 'Z':
            hadamard(q, b)
            phase(q, b)
            phase(q, b)
            hadamard(q, b)
        if s[b] == 'x':
            hadamard(q, b)
        if s[b] == 'X':
            hadamard(q, b)
            phase(q, b)
            phase(q, b)
        if s[b] == 'y':
            hadamard(q, b)
            phase(q, b)
        if s[b] == 'Y':
            hadamard(q, b)
            phase(q, b)
            phase(q, b)
            phase(q, b)

def initstae_(pointer_q, n, string_s):
    # Initialize state q to have n qubits, and input specified by s

    q = pointer_q
    s = str(string_s) if (string_s is not None) else ""

    q.n = n
    q.over32 = (q.n >> 5) + 1

    q.x = np.zeros((2*q.n + 1, q.over32), dtype=int)
    q.z = np.zeros((2*q.n + 1, q.over32), dtype=int)
    q.r = np.zeros((2*q.n + 1, q.over32), dtype=int)

    q.pw[0] = 1
    for i in range(1, 32):
        q.pw[i] = 2 * q.pw[i-1]

    for i in range (2*q.n + 1):
        for j in range(q.over32):
            q.x[i][j] = 0
            q.z[i][j] = 0

        if i < q.n:
            q.x[i][i >> 5] = q.pw[i & 31]

        elif i < 2 * q.n:
            j = i - q.n
            q.z[i][j >> 5] = q.pw[j & 31]

        q.r[i] = 0

    if len(s):
        preparestate(q, s)


def readprog(pointer_h, string_fn, string_params):
    """
    Read the contents of a file as a CHP program
    :param pointer_h: 
    :param string_fn: file name
    :param string_params: list of parameters
    :return: 
    """
    h = pointer_h

    h.DISPQSTATE    = False
    h.DISPTIME      = False
    h.SILENT        = False
    h.DISPPROG      = False
    h.SUPPRESSM     = False

    if (string_params is not None):
        l = len(string_params)
        for t in range(1, l):
            if string_params[t].lower() == 'q':
                h.DISPQSTATE = True
            if string_params[t].lower() == 'p':
                h.DISPPROG = True
            if string_params[t].lower() == 't':
                h.DISPTIME = True
            if string_params[t].lower() == 's':
                h.SILENT = True
            if string_params[t].lower() == 'm':
                h.SUPPRESSM = True

    # number of gates
    h.T = 0

    # number of qubits
    h.n = 0

    with open(string_fn) as fp:
        cnt = 0
        found_end_of_comments = False

        for line in fp:

            # All the lines until the first "#" is encountered are comments in the header
            # Skip them

            if not found_end_of_comments:
                if line.strip() == "#":
                    found_end_of_comments = True
                continue

            l2 = line.strip().split()
            if len(l2) != 0:

                # the gate type
                c = l2[0].lower()

                # the qubit number
                val = int(l2[1])

                if val + 1 > h.n:
                    h.n = val + 1

                if c == 'c':
                    h.a.append(CNOT)
                if c == 'h':
                    h.a.append(HADAMARD)
                if c == 'p':
                    h.a.append(PHASE)
                if c == 'm':
                    h.a.append(MEASURE)

                # first qubit
                h.b.append(int(val))

                if (c == CNOT):
                    # cnots have two qubits
                    h.c.append(int(l2[2]))
                else:
                    # -1 is for nothing
                    h.c.append(-1)

            # print("line {} contents {}".format(cnt, line))
            # cnt += 1

        if not found_end_of_comments:
            error(2)

    # the number of gates is equal to the length
    # of the list where the gates are stored
    h.T = len(h.a)


def main():
    print("pyCHP by Alexandru Paler")
    print("is a translation of")
    print("  CHP: Efficient Simulator for Stabilizer Quantum Circuits")
    print("  by Scott Aaronson")

    # seed random
    seed_value = time.time()
    # for testing purposes force seed_value
    # seed_value = 0
    random.seed(seed_value)

    # allocate h and q
    h = QProg()
    q = QState()

    param = False # whether there are command-line parameters
#
#     srand(time(0))

    argc = len(sys.argv)

    if argc == 1:
        # no parameters specified
        error(0)

    if sys.argv[1][0] == '-':
        # if the parameters str
        param = True

    if param:
        readprog(h, sys.argv[2], sys.argv[1])
    else:
        readprog(h, sys.argv[1], None)

    if argc == (3 + int(param)):
        initstae_(q, h.n, sys.argv[2 + int(param)])
    else:
        initstae_(q, h.n, None)

    runprog(h, q)

if __name__ == "__main__":
    main()