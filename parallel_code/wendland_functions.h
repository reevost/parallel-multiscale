#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef PARALLEL_MULTISCALE_APPROXIMATION_WENDLAND_FUNCTIONS_H
#define PARALLEL_MULTISCALE_APPROXIMATION_WENDLAND_FUNCTIONS_H

unsigned int prod(unsigned int end, unsigned int begin){
    if (begin == end){
        return 1;
    }
    // here I made the splitting to avoid problems if mistakenly the number are exchanged.
    if (begin < end){
        unsigned int cumulative_prod = 1;
        for (unsigned int prod_index = end; prod_index > begin; prod_index--){
            cumulative_prod *= prod_index;
        }
        return cumulative_prod;
    }
    else {
        unsigned int cumulative_prod = 1;
        for (unsigned int prod_index = begin; prod_index > end; prod_index--) {
            cumulative_prod *= prod_index;
        }
        return cumulative_prod;
    }
}

double square_bracket_operator(int alpha, int grade) {
    if (grade == -1){
        return 1.0/(alpha+1);
    }
    if (grade == 0){
        return 1;
    }
    else{
        return prod(alpha, alpha-grade);
    }
}

double curly_bracket_operator(int nu, int grade){
    if (grade == 0){
        return 1;
    }
    if (nu == 0){
        //error handling
        return 0;
    }
    else{
        return prod(nu+grade-1, nu-1);
    }

}

double beta_for_wendland_function(int j, int k_plus_one, int nu){
    if (j == 0 && k_plus_one == 0){
        return 1;
    }
    else{
        double coefficient_sum = 0;
        int k = k_plus_one -1;
        if (j-1 < 0){
            for (int n = 0; n < k_plus_one ; n++) {
                coefficient_sum += beta_for_wendland_function(n, k, nu) * square_bracket_operator(n+1, n-j+1)/
                                   curly_bracket_operator(nu+2*k-n+1, n-j+2);
            }
            return coefficient_sum;
        }
        else{
            for (int n = j-1; n < k_plus_one ; n++) {
                coefficient_sum += beta_for_wendland_function(n, k, nu) * square_bracket_operator(n+1, n-j+1)/
                                   curly_bracket_operator(nu+2*k-n+1, n-j+2);
            }
            return coefficient_sum;
        }
    }
}

// TO DO: add integration part
double wendland_function(double r, double k, int d) {
    // r "points" where the function is evaluated. r is supposed to be ||x-y|| i.e. the norm of some point difference, so r in R^+.
    // k is degree of the function, who is C^2k.
    // d is the dimension of the embedding space.
    if (k == (int) k) {
        int nu = d/2 + k + 1; // optimal value for the purpose of the class function. NOLINT(cppcoreguidelines-narrowing-conversions,bugprone-integer-division)
        double progress_evaluation = 0;
        for (int n = 0; n < k + 1; n++) {
            progress_evaluation += beta_for_wendland_function(n, k, nu) * pow(r, n) * pow(fmax(1 - r, 0), (nu + 2 * k + 1)); // NOLINT(cppcoreguidelines-narrowing-conversions,bugprone-integer-division)
        }
        return progress_evaluation;
    }
    else {
        // we expect even dimension here
        double estimate_integration_value;
        int nu = d/2 + k + 1.0/2; //NOLINT(cppcoreguidelines-narrowing-conversions,bugprone-integer-division)
        if (1 > r && r >= 0) {
            double result = 0; // integrate.quad(lambda t: t * (1 - t) * *nu * (t * *2 - r * *2) * *(k - 1) / (special.gamma(k) * 2 * *(k - 1)), r, 1)
            estimate_integration_value = result;
        }
        else {
            estimate_integration_value = 0;
        }
        return estimate_integration_value;
    }
}

#endif //PARALLEL_MULTISCALE_APPROXIMATION_WENDLAND_FUNCTIONS_H
