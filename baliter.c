#include<stdio.h>
#define MAX 25

int main()
{
	int tab[MAX];
	int j,n,t=0;
	printf("Enter length of array\n");
	scanf("%d",&n);
	printf("Enter 0's and 1's one by one:\n");
	for(j=0;j<n;j++)
		scanf("%d",&tab[j]);
	j=0;
	while (t>=0 && j<n)
	{
		if(tab[j] == 0)
			t+=1;
		else if (tab[j] == 1)
			t-=1;
	}
	if(t<0 || t>0)
		printf("The given sequence of 0's and 1's is not balanced\n");
	else if(t==0)
		printf("The given sequence of 0's and 1's is balanced\n");
	return 0;
}