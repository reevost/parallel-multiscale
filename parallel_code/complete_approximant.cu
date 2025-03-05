#include "macros.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

int main(int argc, char * argv[]){
    /// Expecting parameters are (ordered): Number of levels.
    // arguments definition
    unsigned int number_of_levels;
    // arguments handling and allocation
    if (argc != 2){
        fprintf(stderr, "Expecting 1 parameter, had %d.\n", argc-1);
        return EXIT_FAILURE;
    } else {
        number_of_levels = atoi(argv[1]);
    }

    double data_values[EVALUATION_POINTS_ON_AXIS*EVALUATION_POINTS_ON_AXIS], data_error[EVALUATION_POINTS_ON_AXIS*EVALUATION_POINTS_ON_AXIS];

    /// Set up the loading/storing path
    const char sourceFilePath[200] = __FILE__; // get the path to this file
    char * dirPath; // temporary string to store the current directory path
    char dir_path_domain[200]; // path from the directory of this file to the directory where the results will be stored

    // Find the last occurrence of the directory separator character ("/" or "\")
    const char* lastSeparator = strrchr(sourceFilePath, '/'); // For Unix-like systems
    if (lastSeparator) {
        // Calculate the length of the directory path
        size_t dirPathLength = lastSeparator - sourceFilePath; // find the length of the path to the directory of this file
        dirPath = (char *) malloc((dirPathLength+10)*sizeof(char)); // allocate memory for the directory path
        strncpy(dirPath, sourceFilePath, dirPathLength); // copy the path till the directory
        dirPath[dirPathLength] = '\0'; // add end string
        snprintf(dir_path_domain, 200, "/%dD_domain", POINTS_DIM); // define the path to the storing directory
        strcat(dirPath, dir_path_domain); // append the path to the storing directory to the path to the file directory
        strncpy(dir_path_domain, dirPath, dirPathLength+10); // copy dirPathLength+10 characters from dirPath to dir_path_domain
        // a good remark is that here we cannot directly use dirPath because is defined inside the if-else, we need to reference to an external pointer, in this case dir_path_domain.
        dir_path_domain[dirPathLength+10] = '\0'; //enforce the string termination
        free(dirPath);
        // Print the resulting directory path
        printf("Loading directory path: %s\n", dir_path_domain);
    } else {
        printf("Failed to determine directory path: %s.\n", sourceFilePath);
        return 0;
    }

    double threshold = 1.0/(EVALUATION_POINTS_ON_AXIS-1); // the evaluation points are equispaced points on the domain with "threshold" as grid step.
    // In the following we assume that our domain is within [0,1]^d.
    // We evaluate the approximation on an equispaced grid of EVALUATION_POINTS_ON_AXIS points along each axis for a total of EVALUATION_POINTS_ON_AXIS^POINTS_DIM points    // If we assume that our domain is within [0,1]^d, then

    double temp_grid_point[POINTS_DIM]; // definition per the point of our evaluation grid, to spare storage space we will go through them one by one
    for (int i = 0; i < POINTS_DIM; i++) { temp_grid_point[i] = 0; } // set upo initial values of the grid point, based in our "unitary box" settings
    for (int i = 0; i < pow(EVALUATION_POINTS_ON_AXIS, POINTS_DIM); i++) { // here we assumed still the unitary box settings, i.e. in every dimension there are the same number of evaluation points. Can be generalized.

        // set starting values for the error and the approximant
        data_values[i] = 0;
        data_error[i] = 0.75*exp(-pow(9*temp_grid_point[0]-2, 2)/4-pow(9*temp_grid_point[1]-2, 2)/4)+0.75*exp(-pow(9*temp_grid_point[0]+1,2)/49-pow(9*temp_grid_point[1]+1,2)/49)+0.5*exp(-pow(9*temp_grid_point[0]-7, 2)/4-pow(9*temp_grid_point[1]-3,2)/4)-0.2*exp(-pow(9*temp_grid_point[0]-4,2)-pow(9*temp_grid_point[1]-7,2));

        // increase the grid by one step
        temp_grid_point[0] += threshold;
        for (int d = 1; d < POINTS_DIM; d++) {
            if ((i + 1) % (int) pow(EVALUATION_POINTS_ON_AXIS, d) == 0) { // reached the end along axis "d"
                temp_grid_point[d] += threshold;
                if (d != 0) {
                    temp_grid_point[d - 1] = 0; // based in our "unitary box" settings, should be min along dim "d-1"
                }
            }
        } // end increase
    }

    clock_t save_time = clock();
    for (int level = 0; level < number_of_levels; ++level) {
        // create the file for data storage
        FILE *evaluated_file_pointer; FILE *evaluated_sum_file_pointer; FILE *evaluated_err_file_pointer;
        size_t length = strlen(dir_path_domain);
        char * storage_path = (char*) malloc(length + 200 * sizeof(char));
        char * storage_sum_path = (char*) malloc(length + 200 * sizeof(char));
        char * storage_err_path = (char*) malloc(length + 200 * sizeof(char));
        strcpy(storage_path, dir_path_domain);
        strcpy(storage_sum_path, dir_path_domain);
        strcpy(storage_err_path, dir_path_domain);
        char path[200], sum_path[200], err_path[200];
        snprintf(path, 200, "/result_in_%d_points/approximation%d_%d_evaluated_in_domain_%dD.csv", (int) pow(EVALUATION_POINTS_ON_AXIS, POINTS_DIM), EVALUATION_POINTS_ON_AXIS, level+1, POINTS_DIM);
        snprintf(sum_path, 200, "/result_in_%d_points/approximant%d_%d_evaluated_in_domain_%dD.csv", (int) pow(EVALUATION_POINTS_ON_AXIS, POINTS_DIM), EVALUATION_POINTS_ON_AXIS, level+1, POINTS_DIM);
        snprintf(err_path, 200, "/result_in_%d_points/approximation_error%d_%d_evaluated_in_domain_%dD.csv", (int) pow(EVALUATION_POINTS_ON_AXIS, POINTS_DIM), EVALUATION_POINTS_ON_AXIS, level+1, POINTS_DIM);
        strcat(storage_path, path);
        strcat(storage_sum_path, sum_path);
        strcat(storage_err_path, err_path);
        evaluated_file_pointer = fopen(storage_path, "r");
        double approximation_data[EVALUATION_POINTS_ON_AXIS*EVALUATION_POINTS_ON_AXIS*(POINTS_DIM+1)];
        char data_line[100];
        int index = 0, temp_dim; char * token;
        // read values from file and store them in the matrix pointed by the data_pointer.
        while (fgets(data_line, sizeof data_line, evaluated_file_pointer))
        {
            token = strtok(data_line, ",");
            temp_dim = 0;
            while (token != NULL)
            {
                approximation_data[index + temp_dim * (int) pow(EVALUATION_POINTS_ON_AXIS, POINTS_DIM)] = strtod(token, NULL);
                // printf("dd %f, dim %d, index %d\n", data[index][temp_dim], temp_dim, index);
                token = strtok(NULL, ",");
                temp_dim ++;
            }
            index ++;
        }
        fclose(evaluated_file_pointer);
        evaluated_sum_file_pointer = fopen(storage_sum_path, "w");
        evaluated_err_file_pointer = fopen(storage_err_path, "w");

        for (int i = 0; i < POINTS_DIM; i++) { temp_grid_point[i] = 0; } // set upo initial values of the grid point, based in our "unitary box" settings
        for (int i = 0; i < pow(EVALUATION_POINTS_ON_AXIS, POINTS_DIM); i++) { // here we assumed still the unitary box settings, i.e. in every dimension there are the same number of evaluation points. Can be generalized.
            for (int d = 0; d < POINTS_DIM; d++) {
                fprintf(evaluated_sum_file_pointer, "%f,", temp_grid_point[d]);
                fprintf(evaluated_err_file_pointer, "%f,", temp_grid_point[d]);
            }
            // update error and approximant
            //printf("data: %d-%f\n", i, approx_values[i]);
            data_values[i] += approximation_data[POINTS_DIM*EVALUATION_POINTS_ON_AXIS*EVALUATION_POINTS_ON_AXIS+i];
            data_error[i] -= approximation_data[POINTS_DIM*EVALUATION_POINTS_ON_AXIS*EVALUATION_POINTS_ON_AXIS+i];

            fprintf(evaluated_sum_file_pointer, "%f\n", data_values[i]);
            fprintf(evaluated_err_file_pointer, "%f\n", data_error[i]);

            // increase the grid by one step
            temp_grid_point[0] += threshold;
            for (int d = 1; d < POINTS_DIM; d++) {
                if ((i + 1) % (int) pow(EVALUATION_POINTS_ON_AXIS, d) == 0) { // reached the end along axis "d"
                    temp_grid_point[d] += threshold;
                    if (d != 0) {
                        temp_grid_point[d - 1] = 0; // based in our "unitary box" settings, should be min along dim "d-1"
                    }
                }
            } // end increase
        }
        fclose(evaluated_sum_file_pointer);
        fclose(evaluated_err_file_pointer);
        free(storage_path); free(storage_sum_path); free(storage_err_path);
        clock_t save_time_delta = clock()- save_time;
        printf("Level %d approximation and error saved at time: %f sec.\n", level+1, (double) save_time_delta/CLOCKS_PER_SEC);
    }
}
