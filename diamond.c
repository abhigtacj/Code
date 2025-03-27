#include <stdio.h>

int main() {
    printf("Enter an odd integer\n");
    int n, i, j, k, l, m, val;
    scanf("%d", &n);

    // Check if the input is an odd integer
    if (n % 2 == 0) {
        printf("Incorrect input\n");
        return 0;
    }

    int d = n / 2;

    // Print the upper part of the diamond
    for (i = 1, k = 1, m = d; i <= d; i++, k += 2, m--) {
        for (j = m; j > 0; j--)
            printf("  ");
        for (l = 1; l <= k; l++) {
            printf("* ");
            val = k;
        }
        printf("\n");
    }

    // Print the middle line of the diamond
    for (i = 1; i <= n; i++)
        printf("* ");
    printf("\n");

    // Print the lower part of the diamond
    for (i = d, k = val, m = 1; i > 0; i--, k -= 2, m++) {
        for (j = 1; j <= m; j++)
            printf("  ");
        for (l = k; l > 0; l--)
            printf("* ");
        printf("\n");
    }

    return 0;
}