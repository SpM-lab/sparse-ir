#include "sparseir/sparseir.hpp"
#include "_util.hpp"
#include <memory>

class _AbstractFuncs {
public:
    virtual ~_AbstractFuncs() = default;
    virtual int size() const = 0;
    virtual bool is_continuous_funcs() const = 0;
    virtual std::shared_ptr<_AbstractFuncs> slice(const std::vector<size_t> &indices) const = 0;
};

class AbstractMatsubaraFunctions : public _AbstractFuncs {
public:
    virtual ~AbstractMatsubaraFunctions() = default;
    virtual Eigen::MatrixXcd
    operator()(const Eigen::Array<int64_t, Eigen::Dynamic, 1> &n_array) const = 0;
    virtual bool is_continuous_funcs() const override { return false; }
};

template <typename InternalType>
class MatsubaraBasisFunctions : public AbstractMatsubaraFunctions {
private:
    std::shared_ptr<InternalType> impl;

public:
    MatsubaraBasisFunctions(std::shared_ptr<InternalType> impl) : impl(impl) { }

    virtual Eigen::MatrixXcd
    operator()(const Eigen::Array<int64_t, Eigen::Dynamic, 1> &n_array) const override
    {
        return impl->operator()(n_array);
    }

    virtual int size() const override { return impl->size(); }

    virtual std::shared_ptr<_AbstractFuncs> slice(const std::vector<size_t> &indices) const override
    {
        return _safe_dynamic_pointer_cast<_AbstractFuncs>(std::make_shared<MatsubaraBasisFunctions<InternalType>>(impl->slice(indices)));
    }
};

// Abstract class for functions of a single variable (in the imaginary time
// domain or the real frequency domain)
class AbstractContinuousFunctions : public _AbstractFuncs {
public:
    virtual ~AbstractContinuousFunctions() = default;
    virtual Eigen::VectorXd operator()(double x) const = 0;
    // Returns a matrix of size n_funcs x n_points
    virtual Eigen::MatrixXd operator()(const Eigen::VectorXd &xs) const = 0;
    virtual std::pair<double, double> get_domain() const = 0;
    virtual bool is_continuous_funcs() const override { return true; }

    // Returns the number of roots of the functions
    virtual int nroots() const = 0;

    // Returns the roots of the functions in non-ascending order
    virtual Eigen::VectorXd roots() const = 0;
};

class PiecewiseLegendrePolyFunctions : public AbstractContinuousFunctions {
private:
    std::shared_ptr<sparseir::PiecewiseLegendrePolyVector> impl;

public:
    PiecewiseLegendrePolyFunctions(std::shared_ptr<sparseir::PiecewiseLegendrePolyVector> impl) : impl(impl) { }

    virtual Eigen::VectorXd operator()(double x) const override
    {
        return impl->operator()(x);
    }

    virtual int size() const override { return impl->size(); }

    virtual Eigen::MatrixXd operator()(const Eigen::VectorXd &xs) const override
    {
        return impl->operator()(xs);
    }

    virtual std::pair<double, double> get_domain() const override
    {
        return std::make_pair(impl->xmin(), impl->xmax());
    }

    virtual std::shared_ptr<_AbstractFuncs> slice(const std::vector<size_t> &indices) const override
    {
        return _safe_dynamic_pointer_cast<_AbstractFuncs>(std::make_shared<PiecewiseLegendrePolyFunctions>(impl->slice(indices)));
    }

    virtual int nroots() const override {
        return impl->nroots();
    }

    virtual Eigen::VectorXd roots() const override {
        return impl->roots();
    }
};

template <typename S>
int _sign()
{
    if (std::is_same<S, sparseir::Fermionic>::value) {
        return -1;
    }
    return 1;
}


template <typename ImplType>
class OmegaFunctionsAdaptor : public AbstractContinuousFunctions {
private:
    std::shared_ptr<ImplType> impl;

public:
    OmegaFunctionsAdaptor(std::shared_ptr<ImplType> impl)
        : impl(impl)
    {
    }

    virtual Eigen::VectorXd operator()(double x) const override
    {
        return impl->operator()(x);
    }

    virtual Eigen::MatrixXd operator()(const Eigen::VectorXd &xs) const override
    {
        return impl->operator()(xs);
    }

    virtual int size() const override { return impl->size(); }

    virtual std::pair<double, double> get_domain() const override
    {
        return impl->get_domain();
    }

    virtual std::shared_ptr<_AbstractFuncs> slice(const std::vector<size_t>& indices) const override {
        // First get the sliced implementation
        auto sliced_impl = impl->slice(indices);

        // Convert the sliced implementation to the correct type
        auto converted_impl = _safe_dynamic_pointer_cast<ImplType>(sliced_impl);
        if (!converted_impl) {
            throw std::runtime_error("Failed to convert sliced implementation to correct type");
        }

        // Create new adapter with the converted implementation
        return std::make_shared<OmegaFunctionsAdaptor<ImplType>>(converted_impl);
    }

    virtual int nroots() const override {
        return impl->nroots();
    }

    virtual Eigen::VectorXd roots() const override {
        return impl->roots();
    }
};


template <typename ImplType>
class TauFunctionsAdaptor : public AbstractContinuousFunctions {
private:
    std::shared_ptr<ImplType> impl;
    double beta;

public:
    TauFunctionsAdaptor(std::shared_ptr<ImplType> impl, double beta)
        : impl(impl), beta(beta)
    {
    }

    virtual Eigen::VectorXd operator()(double x) const override
    {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        return impl->operator()(x);
    }

    virtual Eigen::MatrixXd operator()(const Eigen::VectorXd &xs) const override
    {
        return impl->operator()(xs);
    }

    virtual int size() const override { return impl->size(); }

    virtual std::pair<double, double> get_domain() const override
    {
        return std::make_pair(-beta, beta);
    }

    virtual std::shared_ptr<_AbstractFuncs> slice(const std::vector<size_t> &indices) const override
    {
        return _safe_dynamic_pointer_cast<_AbstractFuncs>(std::make_shared<TauFunctionsAdaptor<ImplType>>(impl->slice(indices), beta));
    }

    virtual int nroots() const override {
        return impl->nroots();
    }

    virtual Eigen::VectorXd roots() const override {
        return impl->roots();
    }
};


class AbstractFiniteTempBasis {
public:
    virtual ~AbstractFiniteTempBasis() = default;
    virtual int size() const = 0;
    virtual double get_beta() const = 0;
    virtual int get_statistics() const = 0;
    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const = 0;
    virtual std::shared_ptr<AbstractContinuousFunctions> get_v() const = 0;
    virtual std::shared_ptr<AbstractMatsubaraFunctions> get_uhat() const = 0;
};




template <typename S>
class _IRBasis : public AbstractFiniteTempBasis {
private:
    std::shared_ptr<sparseir::FiniteTempBasis<S>> impl;

public:
    _IRBasis(std::shared_ptr<sparseir::FiniteTempBasis<S>> impl)
        : impl(impl)
    {
    }

    virtual double get_beta() const override { return impl->get_beta(); }

    virtual int size() const override { return impl->size(); }

    virtual int get_statistics() const override
    {
        if (std::is_same<S, sparseir::Fermionic>::value) {
            return SPIR_STATISTICS_FERMIONIC;
        } else {
            return SPIR_STATISTICS_BOSONIC;
        }
    }

    virtual Eigen::VectorXd s() const {
        return impl->s;
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const override
    {
        std::shared_ptr<sparseir::IRTauFuncsType<S>> u_impl = impl->u;
        auto u_tau_funcs = std::make_shared<TauFunctionsAdaptor<sparseir::IRTauFuncsType<S>>>(u_impl, get_beta());
        return std::static_pointer_cast<AbstractContinuousFunctions>(u_tau_funcs);
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_v() const override
    {
        return std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<PiecewiseLegendrePolyFunctions>(impl->v));
    }

    virtual std::shared_ptr<AbstractMatsubaraFunctions>
    get_uhat() const override
    {
        return std::static_pointer_cast<AbstractMatsubaraFunctions>(
            std::make_shared<MatsubaraBasisFunctions<
                sparseir::PiecewiseLegendreFTVector<S>>>(impl->uhat));
    }

    std::shared_ptr<sparseir::FiniteTempBasis<S>> get_impl() const
    {
        return impl;
    }

    std::vector<double> default_tau_sampling_points() const
    {
        // convert from Eigen::VectorXd to std::vector<double>
        auto sampling = impl->default_tau_sampling_points();
        return std::vector<double>(sampling.data(), sampling.data() + sampling.size());
    }

    std::vector<int64_t> default_matsubara_sampling_points(bool positive_only) const
    {
        bool fence = false;
        int L = size();

        std::vector<sparseir::MatsubaraFreq<S>> matsubara_points = impl->default_matsubara_sampling_points(L, fence, positive_only);
        std::vector<int64_t> points(matsubara_points.size());
        std::transform(
            matsubara_points.begin(), matsubara_points.end(), points.begin(),
        [](const sparseir::MatsubaraFreq<S> &freq) {
            return static_cast<int64_t>(freq.get_n());
        });
        return points;
    }

    std::vector<int64_t> default_matsubara_sampling_points_ext(int n_points, bool positive_only) const
    {
        bool fence = false;

        std::vector<sparseir::MatsubaraFreq<S>> matsubara_points = impl->default_matsubara_sampling_points(n_points, fence, positive_only);
        std::vector<int64_t> points(matsubara_points.size());
        std::transform(
            matsubara_points.begin(), matsubara_points.end(), points.begin(),
        [](const sparseir::MatsubaraFreq<S> &freq) {
            return static_cast<int64_t>(freq.get_n());
        });
        return points;
    }

    std::vector<double> default_omega_sampling_points() const
    {
        // convert from Eigen::VectorXd to std::vector<double>
        auto sampling = impl->default_omega_sampling_points();
        return std::vector<double>(sampling.data(), sampling.data() + sampling.size());
    }

};

class AbstractDLR : public AbstractFiniteTempBasis {
public:
    virtual ~AbstractDLR() = default;
    virtual std::shared_ptr<AbstractContinuousFunctions> get_v() const override {
        throw std::runtime_error("get_v is not implemented for DLR");
    }
    virtual std::vector<double> get_poles() const = 0;
};


template <typename S>
class DLRAdapter: public AbstractDLR {
private:
    std::shared_ptr<sparseir::DiscreteLehmannRepresentation<S>> impl;

public:
    DLRAdapter(std::shared_ptr<sparseir::DiscreteLehmannRepresentation<S>> impl)
        : impl(impl)
    {
    }

    virtual int size() const override {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        return impl->size();
    }

    virtual int get_statistics() const override
    {
        if (std::is_same<S, sparseir::Fermionic>::value) {
            return SPIR_STATISTICS_FERMIONIC;
        } else {
            return SPIR_STATISTICS_BOSONIC;
        }
    }

    virtual double get_beta() const override {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        return impl->get_beta();
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const override
    {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        std::shared_ptr<sparseir::DLRTauFuncsType<S>> u_funcs = impl->u;

        return std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<TauFunctionsAdaptor<sparseir::DLRTauFuncsType<S>>>(u_funcs, get_beta())
        );
    }

    virtual std::shared_ptr<AbstractMatsubaraFunctions>
    get_uhat() const override
    {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        return std::static_pointer_cast<AbstractMatsubaraFunctions>(
            std::make_shared<
                MatsubaraBasisFunctions<sparseir::MatsubaraPoles<S>>>(
                impl->uhat));
    }

    virtual std::vector<double> get_poles() const override
    {
        Eigen::VectorXd eigen_poles = impl->poles;
        return std::vector<double>(eigen_poles.data(),
                                   eigen_poles.data() + eigen_poles.size());
    }

    std::shared_ptr<sparseir::DiscreteLehmannRepresentation<S>> get_impl() const
    {
        return impl;
    }
};
