#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

const int NUM_ROWS = 10; // Number of rows in each table

void generateAndWriteCSV(const std::string& filename) {
    std::ofstream file(filename);

    // Check if file is open
    if (!file.is_open()) {
        std::cout << "Failed to open " << filename << std::endl;
        return;
    }

    // Generate and write data
    for (int i = 0; i < NUM_ROWS; ++i) {
        int rid = i; // Row ID
        int intValue = rand() % 20000; // Random integer value
        intValue-=10000; // Random integer value between -10000 and 10000
        file << rid << "," << intValue << "\n";
    }

    file.close();
}

int main() {
    srand(time(NULL)); // Seed for random number generation

    // Generate and write R table
    generateAndWriteCSV("R_table.csv");

    // Generate and write S table
    generateAndWriteCSV("S_table.csv");

    std::cout << "CSV files generated." << std::endl;

    return 0;
}
