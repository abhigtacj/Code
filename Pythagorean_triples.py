count=0
for a in range(1,101):
    for b in range(a,101):
        for c in range(b,101):
            sum=a**2 + b**2
            if c**2==sum:
                count+=1
                print(a,b,c)
print("Number of triples generated is ",count,"\n")