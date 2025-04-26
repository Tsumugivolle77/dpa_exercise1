# Exercises 1

# Problem 1
The nonparallel version of computing pi by Monte Carlo.
```c
void compute_pi_monte_carlo(int n_steps)
{
    int m = 0;
    unsigned int seed = time(NULL) ^ omp_get_thread_num();
    srand(seed);

    for (int i = 0; i < n_steps; i++)
    {
        double x = (double) rand() / RAND_MAX;
        double y = (double) rand() / RAND_MAX;
        if (x * x + y * y < 1.)
        {
            m += 1;
        }
    }

    double pi = 4.0 * m / n_steps;
    printf("Estimated value of pi = %f\n", pi);
}
```

# Problem 2
Here, I implemented a OpenMP version of Monte Carlo. I used `reduction` to avoid **data race** for `m`.
```c
void compute_pi_monte_carlo_parallel(int n_steps)
{
    int m = 0;
#pragma omp parallel
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num();

#pragma omp for reduction(+:m)
        for (int i = 0; i < n_steps; i++)
        {
            double x = (double) rand_r(&seed) / RAND_MAX;
            double y = (double) rand_r(&seed) / RAND_MAX;
            if (x * x + y * y < 1.)
            {
                m++;
            }
        }
    }

    double pi = 4.0 * m / n_steps;
    printf("Estimated value of pi = %f\n", pi);
}
```

# Test Wall Time Elapsed for Our Programs
To test how good our parallel program performs, we need to measure the **Wall Time** (Time consumed in the real world) of our program instead of CPU Time. For the measurement of this, the recommended way is the Linux/MacOS command `time` or the function `omp_get_wtime` provided by `OpenMP` other than `clock`.

I use the following code in my code for the measurement purpose (use `clock` if you wish to compile without `OpenMP`):
```c
int n_steps = 100000000;
double start = omp_get_wtime();
...Your Code/Function For Parallel Computation
double end = omp_get_wtime();
double time_spent = end - start;
printf("Wall time taken: %f seconds\n", time_spent);
```

The testing result when we set `n_steps=100000000`:
```bash
# Parallel version
Estimated value of pi = 3.141505
Wall time taken: 0.229149 seconds
-----------------------------------
# Nonparallel version
Estimated value of pi = 3.141704
Wall time taken: 1.495120 seconds
```