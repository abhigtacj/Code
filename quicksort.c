//quicksort
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
//forward declaration of partition function
int partition(int A[],int low, int high, int *comp);
//quick sort function
void Quicksort(int A[],int low, int high, int *comp)
{
    if(high>low)
    {
        int i=partition(A,low,high,comp);
        Quicksort(A,low,i-1,comp);
        Quicksort(A,i+1,high,comp);
    }
}
//swap function
void swap(int *a, int *b)
{
    int temp=*a;
    *a=*b;
    *b=temp;
}
//partition function
int partition(int A[],int low, int high, int *comp)
{
    int j,i=low-1;
    int mid=(low+high+1)/2;
    //swap(&A[mid],&A[high]);//for pivot at mid
    srand(time(NULL));
    int random=low+rand()%(high-low+1);
    //swap(&A[random],&A[high]);//for random pivot    
    //swap(&A[low],&A[high]);//for pivot at low
    int pivot=A[high];//for pivot at high
    for(j=low;j<high;j++)
    {
        (*comp)++;
        if(A[j]<=pivot)
        {
            i++;
            swap(&A[i],&A[j]);
        }
    }
    swap(&A[i+1],&A[high]);
    return i+1;
}

int main()
{
    int A1[10],A2[10],A3[10],n,i=0,comp=0;
    FILE *fp;
    fp=fopen("data.txt","r");
    if(fp==NULL)
    {
        printf("File open failed\n");
        exit(0);
    }
    while(fscanf(fp,"%d",&n)!=EOF)
    {
        A1[i]=n;
        ++i;
        if(i==10)
        break;
    }
    for(i=0;i<10;i++)
    {
        printf("%d ",A1[i]);
    }
    printf("\n");
    i=0;
    while(fscanf(fp,"%d",&n)!=EOF)
    {
        A2[i]=n;
        ++i;
        if(i==10)
        break;
    }
    for(i=0;i<10;i++)
    {
        printf("%d ",A2[i]);
    }
    printf("\n");
    i=0;
    while(fscanf(fp,"%d",&n)!=EOF)
    {
        A3[i]=n;
        ++i;
        if(i==10)
        break;
    }
    for(i=0;i<10;i++)
    {
        printf("%d ",A3[i]);
    }
    printf("\n");
    fclose(fp);
    Quicksort(A1,0,9,&comp);
    printf("Sorted array : ");
    for(i=0;i<10;i++)
    {
        printf("%d ",A1[i]);
    }
    printf("\n");
    printf("Number of comparisons : %d\n",comp);
    comp=0;
    Quicksort(A2,0,9,&comp);
    printf("Sorted array : ");
    for(i=0;i<10;i++)
    {
        printf("%d ",A2[i]);
    }
    printf("\n");
    printf("Number of comparisons : %d\n",comp);
    comp=0;
    Quicksort(A3,0,9,&comp);
    printf("Sorted array : ");
    for(i=0;i<10;i++)
    {
        printf("%d ",A3[i]);
    }
    printf("\n");
    printf("Number of comparisons : %d\n",comp);
    comp=0;
    return 0;
}