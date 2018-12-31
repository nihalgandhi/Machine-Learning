
arr = [6,-13,2]
new = []
j=0
i=0
while i<len(arr):
    if arr[i] == -13:
        new[j-1] = arr[i-1]/arr[i+1]
        i = i+2
        j += 1
    else:
        new.append(arr[i])
        j += 1
        i += 1
print(new)