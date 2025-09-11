#pragma once

#include <memory>
#include <stdexcept>

// Safe dynamic cast
template <typename T, typename U>
std::shared_ptr<T> _safe_dynamic_cast(const std::shared_ptr<U>& ptr) {
    auto result = std::dynamic_pointer_cast<T>(ptr);
    if (!result) {
        throw std::runtime_error("Failed to dynamic cast, something went wrong! Report this issue to the developers.");
    }
    return result;
}

// Safe static cast
template <typename T, typename U>
std::shared_ptr<T> _safe_static_cast(const std::shared_ptr<U>& ptr) {
    auto result = std::static_pointer_cast<T>(ptr);
    if (!result) {
        throw std::runtime_error("Failed to static cast, something went wrong! Report this issue to the developers.");
    }
    return result;
}

// Safe dynamic pointer cast
template <typename T, typename U>
std::shared_ptr<T> _safe_dynamic_pointer_cast(const std::shared_ptr<U>& ptr) {
    if (!ptr) {
        throw std::runtime_error("Failed to dynamic pointer cast: input pointer is null");
    }
    auto result = std::dynamic_pointer_cast<T>(ptr);
    if (!result) {
        throw std::runtime_error("Failed to dynamic pointer cast, something went wrong! Report this issue to the developers.");
    }
    return result;
}

// Safe static pointer cast
template <typename T, typename U>
std::shared_ptr<T> _safe_static_pointer_cast(const std::shared_ptr<U>& ptr) {
    auto result = std::static_pointer_cast<T>(ptr);
    if (!result) {
        throw std::runtime_error("Failed to static pointer cast, something went wrong! Report this issue to the developers.");
    }
    return result;
} 



// Helper function to check indices for duplicates and out of range
inline void check_indices(const std::vector<size_t>& indices, size_t max_size) {
    std::set<size_t> unique_indices(indices.begin(), indices.end());
    if (unique_indices.size() != indices.size()) {
        throw std::runtime_error("Duplicate indices are not allowed");
    }
    for (size_t idx : indices) {
        if (idx >= max_size) {
            throw std::runtime_error("Index out of range");
        }
    }
}
