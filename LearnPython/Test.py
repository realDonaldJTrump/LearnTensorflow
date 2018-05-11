import sys
import os
import fileinput

print(sys.argv)
print(sys.platform)
args = sys.argv[1:]
args.reverse()
print(os.sep+os.pathsep)
a = os.urandom(2)

for line in fileinput.input(inplace=True):
    line = line.rstrip()
    num = fileinput.filelineno()
    print('%-4s # %2i' % (line, num))