# Exercises 1

# Problem 1
The nonparallel version of computing pi by Monte Carlo.
```cpp
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
```

Testing runtime by:
```bash
cd build
/usr/bin/time ./parallel_algo_demo
```
which gives:
```
Estimated value of pi = 3.963360
        0.02 real         0.01 user         0.01 sys
```

# Problem 2
Here, I implemented a OpenMP version of Monte Carlo. I used `reduction` to avoid **data race** for `m`.
```cpp
void compute_pi_monte_carlo_parallel(int n_steps) {
    int m = 0;

#pragma omp parallel for reduction(+ : m) num_threads(5)
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
```

```bash
cd build
time parallel_algo_demo
```