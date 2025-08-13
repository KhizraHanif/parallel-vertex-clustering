#pragma once

#include <chrono>
#include <iostream>
#include <string>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

struct Time {
private:
    high_resolution_clock::time_point t1, t2;
    double* pSecondsDouble = nullptr;
    bool printTime;
    std::string title;
public:
    Time(std::string name, bool print = false) : t1(high_resolution_clock::now())
        , printTime(print)
        , title(name) {
    }

    Time(std::string name, double* pExecutionTime, bool print = false) : t1(high_resolution_clock::now())
        , pSecondsDouble(pExecutionTime)
        , printTime(print)
        , title(name) {
    }
    ~Time() {
        t2 = high_resolution_clock::now();
        auto const secondsDouble = (t2 - t1).count() * 1e-9;
        if (pSecondsDouble) {
            *pSecondsDouble = secondsDouble;
        }
        if (printTime) {
            std::cout << title << " took " << secondsDouble << " seconds\n";
        }
    }
};
