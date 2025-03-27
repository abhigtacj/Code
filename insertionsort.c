//insertion sort
#include<stdio.h>
#include<stdlib.h>
//insertion sort function
void insertionsort(int Array[],int n)
{
    int i, count=0;
    for(i=1;i<n;i++)
    {
        int key=Array[i];
        int j=i-1;
        while(j>=0 && Array[j]>key)//comparing the elements
        {
            Array[j+1]=Array[j];//swapping the elements
            j=j-1;
            count++;
        }
        Array[j+1]=key;
    }
    printf("The sorted array is : \n");
    for(i=0;i<n;i++)
    {
        printf("%d ",Array[i]);
    }
    printf("\n");
    printf("The total number of comparisons are : %d\n",count);
}
int main()
{
    int A1[10],A2[10],A3[10],n,i=0;
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
    insertionsort(A1,10);
    printf("\n");
    insertionsort(A2,10);
    printf("\n");
    insertionsort(A3,10);
    printf("\n");
    return 0;
}