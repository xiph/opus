#include <string.h>
#include <stdio.h>

int main()
{
	int comment = 0;
	int col = 0;
	char c0, c1;
	c0 = getchar();
	while (!feof(stdin))
	{
		c1 = getchar();
		if (c1==9)
			c1 = 32;
		if (col < 71 || c0 == 10) {
			putchar(c0);
		} else {
			if (c1 == 10 || c1 == 13)
			{
				putchar(c0);
			} else {
				putchar ('\\');
				/*printf ("%d %d %d", col, c0, c1);*/
				putchar (10);
				putchar (c0);
				col=0;
			}
		}
		col++;
		if (c0 == 10)
			col=0;
		c0 = c1;
	}
}