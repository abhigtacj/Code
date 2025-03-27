#Towers of Hanoi
n=3
print("Enter number of disks")
n=int(input())
count=[0]
def towerOfHanoi(n, fromRod, toRod, auxRod):
    if(n!=0):
       towerOfHanoi(n-1, fromRod, auxRod, toRod)
       print("Move disk ",n," from rod ",fromRod," to rod ",toRod,"\n")
       count[0]+=1
       towerOfHanoi(n-1, auxRod, toRod, fromRod)
towerOfHanoi(n,'A','B','C')
print("Total number of shifts is ",count[0],"\n")