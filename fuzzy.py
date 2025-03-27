import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A=[(1,0.8),(2,0.7),(3,0.1),(4,1.0),(5,0.9),(6,0.3)]
B=[(1,0.3),(2,0.4),(3,0.4),(4,1.0),(5,0.2),(6,0.5)]

def alpha_cut(alpha,set):
    return[x for x, mu in set if mu>=alpha]
print("0.5 cut of A:",alpha_cut(0.5,A))
print("0.5 cut of B:",alpha_cut(0.5,B))

def support(set):
    return[x for x, mu in set if mu>0]
print("Support of A:", support(A))
print("Support of B:", support(B))

def scalar_cardinality(set):
    return[mu for _,mu in set]
print("Scalar cardinality of A:",sum(scalar_cardinality(A)))
print("Scalar cardinality of B:",sum(scalar_cardinality(B)))

def core(set):
    return[x for x,mu in set if mu==1]
print("Core of A:",core(A))
print("Core of B:",core(B))

def height(set):
    return[max(mu for _,mu in set)]
print("Height of A:",height(A))
print("Height of B:",height(B))

def subset(set1, set2):
    return (all(any(a==b and mu_a<=mu_b for b,mu_b in set2) for a ,mu_a in set1))
print("Is A subset of B ?",subset(A,B))
print("Is B subset of A ?",subset(B,A))

def complement(set):
    return[(x,1-mu)for x,mu in set]
print("Complement of A:",complement(A))
print("Complement of B:",complement(B))

def union(set1,set2):
    return[(a,max(mu_a,mu_b)) for (a,mu_a),(b,mu_b) in zip(set1,set2)]
print("Union :",union(A,B))

def intersection(set1,set2):
    return[(a,min(mu_a,mu_b)) for (a,mu_a),(b,mu_b) in zip(set1,set2)]
print("Intersection :",intersection(A,B))

def algebraic_sum(set1,set2):
    return[(a,mu_a+mu_b-(mu_a*mu_b)) for (a,mu_a),(b,mu_b) in zip(set1,set2)]
print("Algebraic sum:",algebraic_sum(A,B))

def algebraic_product(set1,set2):
    return[(a,mu_a*mu_b) for (a,mu_a),(b,mu_b) in zip(set1,set2)]
print("Algebraic product:",algebraic_product(A,B))

def power(set,A):
    return[(x,mu**A)for x,mu in set]
print(power(A,2))
print(power(A,1/2))
print(power(B,2))
print(power(B,1/2))

def cartesian_product(set1,set2):
    return[((a,b),min(mu_a,mu_b)) for a,mu_a in set1 for b,mu_b in set2]
print("Cartesian product :",cartesian_product(A,B))

def relation(set1,set2,n):
    return[((a,b),min(mu_a,mu_b)*n) for a,mu_a in set1 for b,mu_b in set2]
print("Relation :",relation(A,B,0.9))

def max_min_composition(set1, set2):
    result = []
    for a, mu_a in set1:
        row = []
        for b, mu_b in set2:
            row.append(min(mu_a, mu_b))
        result.append((a, max(row)))
    return result
print("Max-min composition:", max_min_composition(cartesian_product(A,B), relation(A,B,0.9)))

def max_product_composition(set1, set2):
    result = []
    for a, mu_a in set1:
        row = []
        for b, mu_b in set2:
            row.append(mu_a * mu_b)
        result.append((a, max(row)))
    return result
print("Max-product composition matrix:", max_product_composition(cartesian_product(A, B), relation(A, B, 0.9)))

def triangular_mf(x,a,b,c):
    return max(0,min((x-a)/(b-a),(c-x)/(c-b)))

def trapezoidal_mf(x,a,b,c,d):
    return max(0,min((x-a)/(b-a),1,(d-x)/(d-c)))

def gaussian_mf(x,mean,sigma):
    return np.exp(-((x-mean)**2)/(2*sigma**2))

def sigmoid_mf(x,a,c):
    return 1/(1+np.exp(-a*(x-c)))

X=np.linspace(0,10,100)
Y=[triangular_mf(x,0,5,10)for x in X]
plt.plot(X,Y)
plt.title("Triangular Membership Function")
plt.show()

X=np.linspace(0,10,100)
Y=[trapezoidal_mf(x,0,3,7,10)for x in X]
plt.plot(X,Y)
plt.title("Trapezoidal Membership Function")
plt.show()

X=np.linspace(-10,10,100)
Y=[gaussian_mf(x,0,2)for x in X]
plt.plot(X,Y)
plt.title("Gaussian Membership Function")
plt.show()

X=np.linspace(-10,10,100)
Y=[sigmoid_mf(x,1,0) for x in X]
plt.plot(X,Y)
plt.title("Sigmoid Membership Function")
plt.show()

def generate_random_fuzzy_set(n):
    elements = np.arange(1, n+1)
    membership_values = np.random.rand(n)
    return list(zip(elements, membership_values))
n=int(input('Enter number:'))
random_fuzzy_set=generate_random_fuzzy_set(n)
print(f"Random fuzzy set with {n} elements:", random_fuzzy_set)
print("0.5 cut of random fuzzy set:", alpha_cut(0.5, random_fuzzy_set))
print("Scalar cardinality of random fuzzy set:", sum(scalar_cardinality(random_fuzzy_set)))
print("Complement of random fuzzy set:", complement(random_fuzzy_set))
print("Union of random fuzzy set:", union(random_fuzzy_set,A))
print("Intersection of random fuzzy set:", intersection(random_fuzzy_set,B))