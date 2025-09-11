#pragma once

#include <sparseir/sparseir.hpp>
#include <map>
#include <string>
#include <memory>
#include <chrono>
#include <iostream>

// Global cache for SVE results to avoid recomputing them across all test files
class SVECache {
private:
    // Static cache that will be initialized on first use
    static std::map<std::string, sparseir::SVEResult> &get_cache()
    {
        static std::map<std::string, sparseir::SVEResult> sve_cache;
        return sve_cache;
    }

public:
    // Get SVE result for LogisticKernel
    static sparseir::SVEResult
    get_sve_result(const sparseir::LogisticKernel &kernel, double epsilon)
    {
        // Create a unique key for this kernel and epsilon
        std::string key = "logistic_" + std::to_string(kernel.lambda_) + "_" +
                          std::to_string(epsilon);

        // Get reference to the cache
        auto &cache = get_cache();

        // Check if we already have this result
        auto it = cache.find(key);
        if (it != cache.end()) {
            std::cout << "Using cached SVE result for " << key << std::endl;
            return it->second;
        }

        // Compute and cache the result
        std::cout << "Computing SVE result for " << key << "... ";
        auto start = std::chrono::high_resolution_clock::now();
        auto result = sparseir::compute_sve(kernel, epsilon);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "done in " << elapsed.count() << " seconds" << std::endl;

        cache[key] = result;
        return result;
    }

    // Get SVE result for RegularizedBoseKernel
    static sparseir::SVEResult
    get_sve_result(const sparseir::RegularizedBoseKernel &kernel,
                   double epsilon)
    {
        // Create a unique key for this kernel and epsilon
        std::string key = "regbose_" + std::to_string(kernel.lambda_) + "_" +
                          std::to_string(epsilon);

        // Get reference to the cache
        auto &cache = get_cache();

        // Check if we already have this result
        auto it = cache.find(key);
        if (it != cache.end()) {
            std::cout << "Using cached SVE result for " << key << std::endl;
            return it->second;
        }

        // Compute and cache the result
        std::cout << "Computing SVE result for " << key << "... ";
        auto start = std::chrono::high_resolution_clock::now();
        auto result = sparseir::compute_sve(kernel, epsilon);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "done in " << elapsed.count() << " seconds" << std::endl;

        cache[key] = result;
        return result;
    }

    // Generic method for any kernel type
    template <typename KernelType>
    static sparseir::SVEResult get_sve_result_generic(const KernelType &kernel,
                                                      double epsilon,
                                                      const std::string &prefix)
    {
        // Create a unique key for this kernel and epsilon
        std::string key = prefix + "_" + std::to_string(kernel.lambda_) + "_" +
                          std::to_string(epsilon);

        // Get reference to the cache
        auto &cache = get_cache();

        // Check if we already have this result
        auto it = cache.find(key);
        if (it != cache.end()) {
            std::cout << "Using cached SVE result for " << key << std::endl;
            return it->second;
        }

        // Compute and cache the result
        std::cout << "Computing SVE result for " << key << "... ";
        auto start = std::chrono::high_resolution_clock::now();
        auto result = sparseir::compute_sve(kernel, epsilon);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "done in " << elapsed.count() << " seconds" << std::endl;

        cache[key] = result;
        return result;
    }
};