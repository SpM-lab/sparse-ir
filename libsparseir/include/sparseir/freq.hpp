#pragma once

#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "sparseir/sparseir-fwd.hpp"
#include "sparseir/utils.hpp"

namespace sparseir {

// Base abstract class for Statistics
class Statistics {
public:
    virtual ~Statistics() = default;
    virtual int64_t zeta() const = 0;
    virtual bool allowed(int64_t n) const = 0;
};

// Fermionic statistics class
class Fermionic : public Statistics {
public:
    inline int64_t zeta() const override { return 1; }
    inline bool allowed(int64_t n) const override { return n % 2 != 0; }
};

// Bosonic statistics class
class Bosonic : public Statistics {
public:
    inline int64_t zeta() const override { return 0; }
    inline bool allowed(int64_t n) const override { return n % 2 == 0; }
};

// Factory function to create Statistics object
inline std::unique_ptr<Statistics> create_statistics(int64_t zeta)
{
    if (zeta == 1)
        return sparseir::util::make_unique<Fermionic>();
    if (zeta == 0)
        return sparseir::util::make_unique<Bosonic>();
    throw std::domain_error("Unknown statistics type");
}

// MatsubaraFreq class template
template <typename S>
class MatsubaraFreq {
    static_assert(std::is_base_of<Statistics, S>::value,
                  "S must be derived from Statistics");

public:
    int64_t n;

    // Default constructor
    inline MatsubaraFreq()
        : n(std::is_same<S, Fermionic>::value ? 1 : 0)
    {
        instance_ = std::make_shared<S>();
    }

    // Constructor
    inline MatsubaraFreq(int64_t n) : n(n)
    {
        static_assert(std::is_same<S, Fermionic>::value ||
                          std::is_same<S, Bosonic>::value,
                      "S must be Fermionic or Bosonic");
        S stat;
        if (!stat.allowed(n)) {
            throw std::domain_error("Frequency is not allowed for this type");
        }
        // Store an instance with a shared pointer
        instance_ = std::make_shared<S>(stat);
    }

    // Compute the real value
    inline double value(double beta) const { return n * M_PI / beta; }

    // Compute the imaginary value
    inline std::complex<double> valueim(double beta) const
    {
        std::complex<double> im(0, 1);
        return im * value(beta);
    }

    // Get the statistics instance
    inline S statistics() const
    {
        return *std::static_pointer_cast<S>(instance_);
    }

    // Get n
    inline int64_t get_n() const { return n; }

    // Add conversion operator to long long
    operator long long() const { return static_cast<long long>(n); }

    // Add comparison operators
    bool operator<(const MatsubaraFreq &other) const { return n < other.n; }

    bool operator==(const MatsubaraFreq &other) const { return n == other.n; }

    bool operator!=(const MatsubaraFreq &other) const
    {
        return !(*this == other);
    }

private:
    std::shared_ptr<Statistics> instance_;
};

// Typedefs for convenience
using BosonicFreq = MatsubaraFreq<Bosonic>;
using FermionicFreq = MatsubaraFreq<Fermionic>;

// Overload operators for MatsubaraFreq
template <typename S1, typename S2>
inline MatsubaraFreq<decltype(std::declval<S1>() + std::declval<S2>())>
operator+(const MatsubaraFreq<S1> &a, const MatsubaraFreq<S2> &b)
{
    return MatsubaraFreq<decltype(a.statistics() + b.statistics())>(a.get_n() +
                                                                    b.get_n());
}

template <typename S1, typename S2>
inline MatsubaraFreq<decltype(std::declval<S1>() + std::declval<S2>())>
operator-(const MatsubaraFreq<S1> &a, const MatsubaraFreq<S2> &b)
{
    return MatsubaraFreq<decltype(a.statistics() + b.statistics())>(a.get_n() -
                                                                    b.get_n());
}

template <typename S>
inline MatsubaraFreq<S> operator+(const MatsubaraFreq<S> &a)
{
    return a;
}

template <typename S>
inline MatsubaraFreq<S> operator-(const MatsubaraFreq<S> &a)
{
    return MatsubaraFreq<S>(-a.get_n());
}

inline BosonicFreq operator*(const BosonicFreq &a, int64_t c)
{
    return BosonicFreq(a.get_n() * c);
}

inline FermionicFreq operator*(const FermionicFreq &a, int64_t c)
{
    return FermionicFreq(a.get_n() * c);
}

// Utility functions
template <typename S>
inline int sign(const MatsubaraFreq<S> &a)
{
    return (a.get_n() > 0) - (a.get_n() < 0);
}

template <typename S>
int fermionic_sign()
{
    if (std::is_same<S, sparseir::Fermionic>::value) {
        return -1;
    }
    return 1;
}


template <typename S>
inline BosonicFreq zero()
{
    return BosonicFreq(0);
}

inline bool is_zero(const FermionicFreq &) { return false; }
inline bool is_zero(const BosonicFreq &a) { return a.get_n() == 0; }

template <typename S1, typename S2>
inline bool is_less(const MatsubaraFreq<S1> &a, const MatsubaraFreq<S2> &b)
{
    return a.get_n() < b.get_n();
}

// Custom exception for incompatible promotions
template <typename T1, typename T2>
inline void promote_rule()
{
    throw std::invalid_argument("Will not promote between MatsubaraFreq and "
                                "another type. Use explicit conversion.");
}

// Display function for MatsubaraFreq
template <typename S>
inline void show(std::ostream &os, const MatsubaraFreq<S> &a)
{
    if (a.get_n() == 0)
        os << "0";
    else if (a.get_n() == 1)
        os << "π/β";
    else if (a.get_n() == -1)
        os << "-π/β";
    else
        os << a.get_n() << "π/β";
}

} // namespace sparseir