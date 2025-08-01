def mcp_neuron(inputs, weights, threshold):
    summation = sum(i * w for i, w in zip(inputs, weights)) 
    if summation >= threshold:
        return 1
    else:
        return 0

def AND(x1, x2):
    return mcp_neuron([x1, x2], [1, 1], 2)

def OR(x1, x2):
    return mcp_neuron([x1, x2], [1, 1], 1)

def NOT(x1):
    return mcp_neuron([x1], [-1], 0)

def NOR(x1, x2):
    return mcp_neuron([x1, x2], [-1, -1], 0)

def XOR(x1, x2):
    return x1 ^ x2

# Testing all logic gates
print("AND")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {AND(*x)}")

print("\nOR")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {OR(*x)}")

print("\nNOT")
for x in [0, 1]:
    print(f"{x} -> {NOT(x)}")

print("\nNOR")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {NOR(*x)}")

print("\nXOR")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {XOR(*x)}")
