#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void compute_pi_monte_carlo(int n_steps) {
    int m = 0;

    for (int i = 0; i < n_steps; i++)
    {
        unsigned int seed = (unsigned int) clock();
        double x = (double) rand_r(&seed) / RAND_MAX;
        double y = (double) rand_r(&seed) / RAND_MAX;
        if (x * x + y * y < 1.)
        {
            m += 1;
        }
    }

    double pi = 4.0 * m / n_steps;
    printf("Estimated value of pi = %f\n", pi);
}

void compute_pi_monte_carlo_parallel(int n_steps) {
    int m = 0;
#pragma omp parallel for reduction(+ : m)
    for (int i = 0; i < n_steps; i++)
    {
        unsigned int seed = (unsigned int) clock();
        double x = (double) rand_r(&seed) / RAND_MAX;
        double y = (double) rand_r(&seed) / RAND_MAX;
        if (x * x + y * y < 1.)
        {
            m += 1;
        }
    }

    double pi = 4.0 * m / n_steps;
    printf("Estimated value of pi = %f\n", pi);
}

int main()
{
    // compute_pi_monte_carlo(1000000);
    compute_pi_monte_carlo_parallel(1000000);

    return 0;
}